"""Quantum-conditioned report generator (paper Sec. 3.6).

A Q-Former fuses three token streams - pooled encoder visual features F_vis,
graph node tokens / graph embedding g, and the quantum measurement vector z -
into T_vis, T_graph and T_quant (Eq. 15). The query outputs are projected into
the embedding space of a LoRA-adapted language model (Llama 3.2 11B
Vision-Instruct in the paper) and prepended to a chain-of-thought prompt that
enforces the five report sections. Training uses the autoregressive loss
L_txt and the InfoNCE alignment loss L_align (Eq. 16).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

REPORT_SECTIONS = ["Technique", "Findings", "Laterality and Lobe", "Impression", "BT-RADS"]

SYSTEM_PROMPT = (
    "You are an expert neuroradiologist. You write structured brain MRI reports "
    "grounded strictly in the supplied image evidence and measurements."
)

USER_TEMPLATE = (
    "Write a structured radiology report for this multi-parametric brain MRI "
    "({modalities}). Reason step by step: first check the sub-region measurements, "
    "then the anatomical location, then the predicted tumor type, and only then "
    "write the report. The report must contain exactly these sections, in order:\n"
    "Technique:\n"
    "Findings: (per sub-region, volumes in cm³, enhancement and necrosis ratios)\n"
    "Laterality and Lobe: (from the atlas intersection)\n"
    "Impression:\n"
    "BT-RADS: (score and category)\n\n"
    "Measurements: {facts}\n"
    "Predicted tumor type: {tumor_type} (confidence {confidence:.2f})"
)


# --------------------------------------------------------------------------
# Q-Former
# --------------------------------------------------------------------------

class QFormerProjector(nn.Module):
    def __init__(self, vis_dim: int, graph_dim: int, z_dim: int, llm_dim: int,
                 n_vis_queries=32, n_graph_queries=8, n_quant_queries=4,
                 num_layers=4, hidden_size=768, encoder_hidden_size=1408,
                 cross_attention_frequency=1, blip2_init: str | None = None):
        super().__init__()
        from transformers import Blip2QFormerConfig, Blip2QFormerModel

        cfg = Blip2QFormerConfig(hidden_size=hidden_size, num_hidden_layers=num_layers,
                                 num_attention_heads=12, intermediate_size=4 * hidden_size,
                                 encoder_hidden_size=encoder_hidden_size,
                                 cross_attention_frequency=cross_attention_frequency)
        self.qformer = Blip2QFormerModel(cfg)
        self.n_q = (n_vis_queries, n_graph_queries, n_quant_queries)
        self.queries = nn.Parameter(0.02 * torch.randn(1, sum(self.n_q), hidden_size))
        self.in_vis = nn.Linear(vis_dim, encoder_hidden_size)
        self.in_graph = nn.Linear(graph_dim, encoder_hidden_size)
        self.in_z = nn.Linear(z_dim, encoder_hidden_size)
        self.stream_emb = nn.Embedding(3, encoder_hidden_size)
        self.to_llm = nn.Linear(hidden_size, llm_dim)
        if blip2_init:
            load_blip2_qformer(self, blip2_init)

    def forward(self, vis_tokens, vis_mask, graph_tokens, graph_mask, z):
        dev = vis_tokens.device
        ev = self.in_vis(vis_tokens) + self.stream_emb.weight[0]
        eg = self.in_graph(graph_tokens) + self.stream_emb.weight[1]
        ez = self.in_z(z.float()).unsqueeze(1) + self.stream_emb.weight[2]
        enc = torch.cat([ev, eg, ez], 1)
        enc_mask = torch.cat([vis_mask, graph_mask,
                              torch.ones(z.shape[0], 1, dtype=vis_mask.dtype, device=dev)], 1)
        q = self.queries.expand(vis_tokens.shape[0], -1, -1)
        out = self.qformer(query_embeds=q, encoder_hidden_states=enc,
                           encoder_attention_mask=enc_mask.long()).last_hidden_state
        out = self.to_llm(out)
        t_vis, t_graph, t_quant = torch.split(out, list(self.n_q), dim=1)
        return t_vis, t_graph, t_quant


def load_blip2_qformer(proj: QFormerProjector, repo_id: str = "Salesforce/blip2-opt-2.7b"):
    """Copy shape-compatible Q-Former weights (and the 32 query tokens) from a
    BLIP-2 checkpoint on the Hugging Face Hub without loading the full model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        index = hf_hub_download(repo_id, "model.safetensors.index.json")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({v for k, v in weight_map.items()
                         if k.startswith("qformer.") or k == "query_tokens"})
    except Exception:  # single-file checkpoint
        shards = ["model.safetensors"]
    own = proj.qformer.state_dict()
    loaded = 0
    for shard in shards:
        path = hf_hub_download(repo_id, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key == "query_tokens":
                    t = f.get_tensor(key)
                    n = min(t.shape[1], proj.queries.shape[1])
                    with torch.no_grad():
                        proj.queries[:, :n].copy_(t[:, :n])
                    loaded += 1
                elif key.startswith("qformer."):
                    k = key[len("qformer."):]
                    if k in own and own[k].shape == f.get_slice(key).get_shape():
                        own[k] = f.get_tensor(key)
                        loaded += 1
    proj.qformer.load_state_dict(own)
    log.info("Loaded %d BLIP-2 Q-Former tensors from %s", loaded, repo_id)
    return loaded


# --------------------------------------------------------------------------
# Language model wrapper
# --------------------------------------------------------------------------

@dataclass
class LLMConfig:
    name_or_path: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    load_in_4bit: bool = False
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    max_prompt_tokens: int = 384
    max_report_tokens: int = 384
    align_dim: int = 256
    temperature: float = 0.07


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_language_model(cfg: LLMConfig):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if cfg.name_or_path == "toy":
        from .toy_llm import toy_llm

        return toy_llm()

    kwargs = {"dtype": _dtype(cfg.dtype)}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype(cfg.dtype), bnb_4bit_use_double_quant=True)
    hf_cfg = AutoConfig.from_pretrained(cfg.name_or_path)
    if hf_cfg.model_type == "mllama":
        from transformers import MllamaForConditionalGeneration as ModelCls
    else:
        ModelCls = AutoModelForCausalLM
    model = ModelCls.from_pretrained(cfg.name_or_path, **kwargs)
    tok = AutoTokenizer.from_pretrained(cfg.name_or_path)
    return model, tok


def apply_lora(model, cfg: LLMConfig):
    from peft import LoraConfig, get_peft_model

    targets = cfg.lora_target_modules
    if getattr(model.config, "model_type", "") == "mllama":
        # adapt the self-attention projections of the language model only
        targets = r".*language_model.*\.self_attn\.(" + "|".join(targets) + r")"
    lcfg = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=targets,
                      bias="none", task_type="CAUSAL_LM")
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, cfg.gradient_checkpointing)
    return get_peft_model(model, lcfg)


class ReportGenerator(nn.Module):
    def __init__(self, cfg: LLMConfig, model=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        if model is None:
            model, tokenizer = load_language_model(cfg)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for p in model.parameters():
            p.requires_grad_(False)
        if cfg.gradient_checkpointing and not cfg.load_in_4bit:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        self.llm = apply_lora(model, cfg) if cfg.lora_rank > 0 else model
        self.tok = tokenizer
        hidden = self.hidden_size
        self.align_vis = nn.Linear(hidden, cfg.align_dim)
        self.align_txt = nn.Linear(hidden, cfg.align_dim)

    @property
    def hidden_size(self) -> int:
        c = self.llm.config
        c = getattr(c, "text_config", c)
        return c.hidden_size

    # ---------------- prompt handling ----------------
    def build_prompt(self, facts: str, tumor_type: str, confidence: float,
                     modalities: str = "T1, T1-CE, T2, FLAIR") -> str:
        user = USER_TEMPLATE.format(facts=facts, tumor_type=tumor_type,
                                    confidence=confidence, modalities=modalities)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"{SYSTEM_PROMPT}\n\n{user}\n\nReport:\n"

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def _assemble(self, prefix, prompts, reports=None, pad_left=False):
        """Build inputs_embeds / attention mask / labels for a batch."""
        dev = prefix.device
        emb_dtype = self.llm.get_input_embeddings().weight.dtype
        seqs, labels, rep_masks = [], [], []
        for b, prompt in enumerate(prompts):
            p_ids = self.tok(prompt, add_special_tokens=False, truncation=True,
                             max_length=self.cfg.max_prompt_tokens,
                             return_tensors="pt").input_ids[0].to(dev)
            if self.tok.bos_token_id is not None and (len(p_ids) == 0 or p_ids[0] != self.tok.bos_token_id):
                p_ids = torch.cat([torch.tensor([self.tok.bos_token_id], device=dev), p_ids])
            parts = [self._embed(p_ids[:1]), prefix[b].to(emb_dtype), self._embed(p_ids[1:])]
            lab = [torch.full((1 + prefix.shape[1] + len(p_ids) - 1,), -100, device=dev)]
            rmask = [torch.zeros(lab[0].shape[0], dtype=torch.bool, device=dev)]
            if reports is not None:
                r_ids = self.tok(reports[b], add_special_tokens=False, truncation=True,
                                 max_length=self.cfg.max_report_tokens - 1,
                                 return_tensors="pt").input_ids[0].to(dev)
                r_ids = torch.cat([r_ids, torch.tensor([self.tok.eos_token_id], device=dev)])
                parts.append(self._embed(r_ids))
                lab.append(r_ids)
                rmask.append(torch.ones(len(r_ids), dtype=torch.bool, device=dev))
            seqs.append(torch.cat(parts, 0))
            labels.append(torch.cat(lab))
            rep_masks.append(torch.cat(rmask))
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        emb = torch.zeros(len(seqs), L, H, dtype=emb_dtype, device=dev)
        att = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long, device=dev)
        rep = torch.zeros(len(seqs), L, dtype=torch.bool, device=dev)
        for i, s in enumerate(seqs):
            n = s.shape[0]
            sl = slice(L - n, L) if pad_left else slice(0, n)
            emb[i, sl] = s
            att[i, sl] = 1
            lab[i, sl] = labels[i]
            rep[i, sl] = rep_masks[i]
        return emb, att, lab, rep

    # ---------------- training ----------------
    def forward(self, t_vis, t_graph, t_quant, prompts, reports):
        prefix = torch.cat([t_vis, t_graph, t_quant], 1)
        emb, att, lab, rep = self._assemble(prefix, prompts, reports)
        out = self.llm(inputs_embeds=emb, attention_mask=att, labels=lab,
                       output_hidden_states=True, return_dict=True)
        l_txt = out.loss
        hid = out.hidden_states[-1].float()
        txt = (hid * rep.unsqueeze(-1)).sum(1) / rep.sum(1, keepdim=True).clamp_min(1)
        vis = t_vis.float().mean(1)
        l_align = info_nce(self.align_vis(vis), self.align_txt(txt), self.cfg.temperature)
        return l_txt, l_align

    # ---------------- inference ----------------
    @torch.no_grad()
    def generate(self, t_vis, t_graph, t_quant, prompts, max_new_tokens=384, **gen_kwargs):
        prefix = torch.cat([t_vis, t_graph, t_quant], 1)
        emb, att, _, _ = self._assemble(prefix, prompts, None, pad_left=True)
        gen_kwargs.setdefault("do_sample", False)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=att,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=self.tok.pad_token_id, **gen_kwargs)
        return self.tok.batch_decode(out, skip_special_tokens=True)


def info_nce(vis: torch.Tensor, txt: torch.Tensor, temperature=0.07) -> torch.Tensor:
    """L_align = -log exp(s(v_i, t_i)/tau) / sum_j exp(s(v_i, t_j)/tau), cosine s."""
    v = F.normalize(vis, dim=-1)
    t = F.normalize(txt, dim=-1)
    logits = v @ t.T / temperature
    target = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, target)

"""The paper's backbone is Llama 3.2 Vision (Mllama). Check the wrapper against a
tiny randomly initialised Mllama so LoRA targeting and inputs_embeds work."""
import pytest
import torch


def test_report_generator_on_tiny_mllama():
    mllama = pytest.importorskip("transformers.models.mllama.configuration_mllama")
    from transformers import MllamaConfig, MllamaForConditionalGeneration

    from mqda.models.report import LLMConfig, ReportGenerator
    from mqda.models.toy_llm import toy_tokenizer

    tc = mllama.MllamaTextConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                                 num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                 cross_attention_layers=[1, 3], bos_token_id=1, eos_token_id=2,
                                 pad_token_id=0)
    vc = mllama.MllamaVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_global_layers=1, num_attention_heads=4, image_size=56,
                                   patch_size=14, vision_output_dim=64, intermediate_layers_indices=[0])
    torch.manual_seed(0)
    model = MllamaForConditionalGeneration(MllamaConfig(vision_config=vc, text_config=tc))
    tok = toy_tokenizer()
    model.resize_token_embeddings(len(tok))
    rg = ReportGenerator(LLMConfig(name_or_path="tiny-mllama", lora_rank=4, dtype="float32",
                                   gradient_checkpointing=False, align_dim=8), model, tok)
    lora = [n for n, p in rg.named_parameters() if "lora_A" in n]
    # LoRA only on the language model's self-attention layers (0 and 2)
    assert lora and all("language_model" in n and "self_attn" in n for n in lora)
    assert {n.split("layers.")[1].split(".")[0] for n in lora} == {"0", "2"}

    tv, tg, tq = torch.randn(2, 4, 64), torch.randn(2, 2, 64), torch.randn(2, 1, 64)
    l_txt, l_align = rg(tv, tg, tq, ["prompt a", "prompt bb"], ["report one", "report two"])
    (l_txt + l_align).backward()
    assert torch.isfinite(l_txt) and torch.isfinite(l_align)
    rg.eval()
    assert len(rg.generate(tv, tg, tq, ["prompt a", "p"], max_new_tokens=3)) == 2

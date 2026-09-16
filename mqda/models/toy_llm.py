"""Offline toy language model (``llm.name_or_path: toy``) for smoke tests.

Builds a randomly initialised two-layer Llama and a character-level tokenizer
without any download, so the whole joint-training pipeline can be exercised
on a CPU. It produces meaningless text by design.
"""
from __future__ import annotations

import torch

_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;()%=-+/\n³²'")


def toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for c in _CHARS:
        vocab.setdefault(c, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>")


def toy_llm(hidden=64, layers=2, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = toy_tokenizer()
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, bos_token_id=1, eos_token_id=2,
                      pad_token_id=0)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg), tok

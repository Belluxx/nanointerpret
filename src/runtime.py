from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ATTENTION_IMPLEMENTATION = "sdpa"


def load_tokenizer(model_id: str, tokenizer=None):
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(device)
    return model.eval().requires_grad_(False)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
            print("warning: neither MPS nor CUDA is available; using CPU", file=sys.stderr)

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(requested)

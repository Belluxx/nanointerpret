from __future__ import annotations

import sys

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


ATTENTION_IMPLEMENTATION = "sdpa"


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


def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList)
        and name.split(".")[-1] == "layers"
        and len(module) > 1
    ]
    if not candidates:
        raise RuntimeError("could not locate the transformer's ModuleList named 'layers'")
    return max(candidates, key=len)


def compile_transformer_prefix(layers: nn.ModuleList, layer_index: int) -> None:
    # Keep the capture layer eager so its forward pre-hook remains visible.
    for index in range(layer_index):
        layers[index] = torch.compile(layers[index], dynamic=False)


class _ActivationCaptured(Exception):
    pass


@torch.no_grad()
def capture_layer_input(model: nn.Module, layer: nn.Module, input_ids: Tensor) -> Tensor:
    # Run the model only up to `layer` and return its input hidden states.
    activation = None

    def capture(_module, args, kwargs):
        nonlocal activation
        activation = (args[0] if args else kwargs["hidden_states"]).detach()
        raise _ActivationCaptured

    handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(input_ids=input_ids, use_cache=False)
    except _ActivationCaptured:
        pass
    finally:
        handle.remove()
    if activation is None:
        raise RuntimeError("the residual-stream hook did not run")
    return activation

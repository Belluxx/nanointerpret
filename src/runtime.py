from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_causal_lm(model_id: str, dtype: str, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval().requires_grad_(False)


def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] == "layers" and len(module) > 1
    ]
    if not candidates:
        raise RuntimeError("could not locate the transformer's ModuleList named 'layers'")
    return max(candidates, key=len)


@contextmanager
def patch_layer_input(layer: nn.Module, patch: Callable[[Tensor], Tensor]) -> Iterator[None]:
    def hook(_module, args, kwargs):
        if args:
            return (patch(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": patch(kwargs["hidden_states"])}

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class _Captured(Exception):
    pass


@torch.no_grad()
def capture_layer_input(model: nn.Module, layer: nn.Module, input_ids: Tensor) -> Tensor:
    # Run the model only up to `layer` and return its input hidden states.
    def stop(hidden: Tensor) -> Tensor:
        raise _Captured(hidden)

    with patch_layer_input(layer, stop):
        try:
            model(input_ids=input_ids, use_cache=False)
        except _Captured as captured:
            return captured.args[0]
    raise RuntimeError("the residual-stream hook did not run")

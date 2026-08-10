from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .experiment import find_transformer_layers
from .sae import TopKSAE, load_sae


@dataclass(frozen=True)
class InterventionRequest:
    prompt: str
    feature_id: int
    mode: str
    amount: float
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float

    @classmethod
    def from_payload(cls, payload: dict) -> InterventionRequest:
        mode = payload["mode"]
        parameter = "clamp_value" if mode == "clamp" else "alpha"
        return cls(
            payload["prompt"],
            payload["feature_id"],
            mode,
            payload[parameter],
            payload["max_new_tokens"],
            payload["temperature"],
            payload["top_p"],
            payload["top_k"],
            payload["repetition_penalty"],
        )


def _feature_activation(
    sae: TopKSAE, normalized_residual: Tensor, feature_id: int
) -> Tensor:
    shape = normalized_residual.shape[:-1]
    x = normalized_residual.reshape(-1, sae.d_model)
    if sae.subtract_pre_bias:
        x = x - sae.decoder_bias
    pre_activations = x @ sae.encoder_weight + sae.encoder_bias
    values, indices = torch.topk(
        F.relu(pre_activations), sae.k, dim=-1, sorted=False
    )
    activation = (values * (indices == feature_id)).sum(dim=-1)
    return activation.reshape(shape)


def apply_feature_intervention(
    residual: Tensor,
    sae: TopKSAE,
    feature_id: int,
    mode: str,
    amount: float,
    *,
    activation_scale: float = 1.0,
) -> Tensor:
    """Return a residual stream with one SAE decoder-direction edit.

    The SAE operates on ``activation_scale * residual``. The edit is therefore
    calculated in the SAE's normalized coordinates and divided by that scale
    before it is added to the model's residual stream.
    """
    direction = sae.decoder_weight[feature_id]
    if mode == "clamp":
        normalized = residual.to(sae.decoder_weight.dtype) * activation_scale
        current = _feature_activation(sae, normalized, feature_id)
        delta = (amount - current).unsqueeze(-1) * direction
    else:
        delta = amount * direction
    return residual + (delta / activation_scale).to(residual.dtype)


@contextmanager
def feature_intervention_hook(
    layer: nn.Module,
    sae: TopKSAE,
    feature_id: int,
    mode: str,
    amount: float,
    activation_scale: float,
):
    def intervene(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        # Preserve the prompt's cached context; steer its final token and new tokens.
        modified_last = apply_feature_intervention(
            hidden[:, -1:],
            sae,
            feature_id,
            mode,
            amount,
            activation_scale=activation_scale,
        )
        modified = torch.cat((hidden[:, :-1], modified_last), dim=1)
        if args:
            return (modified, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": modified}

    handle = layer.register_forward_pre_hook(intervene, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class InterventionGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer: nn.Module,
        sae: TopKSAE,
        activation_scale: float,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.sae = sae
        self.activation_scale = activation_scale
        self.device = device

    @classmethod
    def from_sae_dir(
        cls,
        sae_dir: Path,
        device: torch.device,
        *,
        tokenizer=None,
    ) -> InterventionGenerator:
        config_path = sae_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SAE configuration not found: {config_path}")
        config = json.loads(config_path.read_text())

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(config["model_id"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=getattr(torch, config["model_dtype"]),
            attn_implementation="eager",
        ).to(device)
        model.eval().requires_grad_(False)

        _layer_path, layers = find_transformer_layers(model)
        layer_index = int(config["layer_index"])
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"configured activation layer {layer_index} is not present in the model"
            )
        sae = load_sae(sae_dir, config, device)
        return cls(
            model,
            tokenizer,
            layers[layer_index],
            sae,
            float(config["activation_scale"]),
            device,
        )

    @torch.inference_mode()
    def generate_pair(self, request: InterventionRequest) -> dict:
        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        generation_args = {
            **inputs,
            "do_sample": request.temperature > 0,
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": request.repetition_penalty,
        }
        if request.temperature > 0:
            generation_args.update(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )

        sampling_seed = torch.seed()
        torch.manual_seed(sampling_seed)
        baseline_ids = self.model.generate(**generation_args)
        torch.manual_seed(sampling_seed)
        with feature_intervention_hook(
            self.layer,
            self.sae,
            request.feature_id,
            request.mode,
            request.amount,
            self.activation_scale,
        ):
            intervened_ids = self.model.generate(**generation_args)

        return {
            "baseline": self._decode_generation(baseline_ids[0], prompt_tokens),
            "intervened": self._decode_generation(intervened_ids[0], prompt_tokens),
        }

    def _decode_generation(self, token_ids: Tensor, prompt_tokens: int) -> str:
        return self.tokenizer.decode(
            token_ids[prompt_tokens:], skip_special_tokens=True
        )

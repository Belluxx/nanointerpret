from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .experiment import find_transformer_layers
from .sae import TopKSAE


MAX_NEW_TOKENS = 256
MAX_PROMPT_CHARACTERS = 20_000


@dataclass(frozen=True)
class InterventionRequest:
    prompt: str
    feature_id: int
    mode: str
    amount: float
    max_new_tokens: int

    @classmethod
    def from_payload(
        cls, payload: object, d_sae: int
    ) -> InterventionRequest:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"prompt must be at most {MAX_PROMPT_CHARACTERS:,} characters"
            )

        feature_id = payload.get("feature_id")
        if isinstance(feature_id, bool) or not isinstance(feature_id, int):
            raise ValueError("feature_id must be an integer")
        if not 0 <= feature_id < d_sae:
            raise ValueError(f"feature_id must be between 0 and {d_sae - 1}")

        mode = payload.get("mode")
        if mode == "clamp":
            parameter = "clamp_value"
        elif mode == "additive":
            parameter = "alpha"
        else:
            raise ValueError("mode must be 'clamp' or 'additive'")

        amount = payload.get(parameter)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"{parameter} must be a finite number")
        amount = float(amount)
        if not math.isfinite(amount):
            raise ValueError(f"{parameter} must be a finite number")

        max_new_tokens = payload.get("max_new_tokens")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(
                f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}"
            )

        return cls(prompt, feature_id, mode, amount, max_new_tokens)

    def intervention(self) -> dict[str, str | float]:
        parameter = "clamp_value" if self.mode == "clamp" else "alpha"
        return {"mode": self.mode, parameter: self.amount}


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
    if residual.shape[-1] != sae.d_model:
        raise ValueError(
            f"expected residual width {sae.d_model}, got {residual.shape[-1]}"
        )
    if not 0 <= feature_id < sae.d_sae:
        raise ValueError(f"feature_id must be between 0 and {sae.d_sae - 1}")
    if not math.isfinite(activation_scale) or activation_scale <= 0:
        raise ValueError("activation_scale must be positive and finite")
    if not math.isfinite(amount):
        raise ValueError("intervention amount must be finite")

    sae_dtype = sae.decoder_weight.dtype
    direction = sae.decoder_weight[feature_id]
    if mode == "clamp":
        normalized = residual.to(dtype=sae_dtype) * activation_scale
        current = _feature_activation(sae, normalized, feature_id)
        coefficient = amount - current
    elif mode == "additive":
        coefficient = torch.full(
            residual.shape[:-1], amount, dtype=sae_dtype, device=residual.device
        )
    else:
        raise ValueError("mode must be 'clamp' or 'additive'")

    normalized_delta = coefficient.unsqueeze(-1) * direction
    residual_delta = (normalized_delta / activation_scale).to(residual.dtype)
    return residual + residual_delta


class FeatureInterventionHook:
    def __init__(
        self,
        layer: nn.Module,
        sae: TopKSAE,
        feature_id: int,
        mode: str,
        amount: float,
        activation_scale: float,
    ):
        def intervene(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            modified = apply_feature_intervention(
                hidden,
                sae,
                feature_id,
                mode,
                amount,
                activation_scale=activation_scale,
            )
            if args:
                return (modified, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": modified}

        self.handle = layer.register_forward_pre_hook(
            intervene, with_kwargs=True
        )

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> FeatureInterventionHook:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_sae(sae_dir: Path, config: dict, device: torch.device) -> TopKSAE:
    checkpoint_path = sae_dir / "sae_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    decoder_weight = checkpoint["sae"]["decoder_weight"]
    d_sae, d_model = decoder_weight.shape
    sae = TopKSAE(
        d_model,
        d_sae,
        int(config["k"]),
        device,
        subtract_pre_bias=bool(config["subtract_pre_bias"]),
    )
    sae.load_state_dict(checkpoint["sae"])
    return sae.eval().requires_grad_(False)


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
        self.lock = threading.Lock()

    @classmethod
    def from_sae_dir(
        cls,
        sae_dir: Path,
        device: torch.device,
        *,
        model_dtype: str | None = None,
        tokenizer=None,
    ) -> InterventionGenerator:
        config_path = sae_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SAE configuration not found: {config_path}")
        config = json.loads(config_path.read_text())

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(config["model_id"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = getattr(torch, model_dtype or config["model_dtype"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], dtype=dtype, attn_implementation="eager"
        ).to(device)
        model.eval().requires_grad_(False)

        _layer_path, layers = find_transformer_layers(model)
        layer_index = int(config["layer_index"])
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"configured activation layer {layer_index} is not present in the model"
            )
        sae = load_sae(sae_dir, config, device)
        if sae.d_model != int(model.config.hidden_size):
            raise ValueError("SAE and model residual widths do not match")
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
        if request.feature_id >= self.sae.d_sae:
            raise ValueError(
                f"feature_id must be between 0 and {self.sae.d_sae - 1}"
            )

        with self.lock:
            inputs = self.tokenizer(request.prompt, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]
            generation_args = {
                **inputs,
                "do_sample": False,
                "min_new_tokens": request.max_new_tokens,
                "max_new_tokens": request.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            baseline_ids = self.model.generate(**generation_args)
            with FeatureInterventionHook(
                self.layer,
                self.sae,
                request.feature_id,
                request.mode,
                request.amount,
                self.activation_scale,
            ):
                intervened_ids = self.model.generate(**generation_args)

        return {
            "prompt": request.prompt,
            "feature_id": request.feature_id,
            "intervention": request.intervention(),
            "max_new_tokens": request.max_new_tokens,
            "baseline": self._decode_generation(baseline_ids[0], prompt_tokens),
            "intervened": self._decode_generation(
                intervened_ids[0], prompt_tokens
            ),
        }

    def _decode_generation(self, token_ids: Tensor, prompt_tokens: int) -> dict:
        continuation_ids = token_ids[prompt_tokens:]
        return {
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
            "continuation": self.tokenizer.decode(
                continuation_ids, skip_special_tokens=True
            ),
            "generated_tokens": len(continuation_ids),
        }

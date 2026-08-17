from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .experiment import find_transformer_layers
from .runtime import load_causal_lm, load_tokenizer
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


def _feature_activation(
    sae: TopKSAE, normalized_residual: Tensor, feature_id: int | Tensor
) -> Tensor:
    shape = normalized_residual.shape[:-1]
    x = normalized_residual.reshape(-1, sae.d_model)
    indices, values = sae.encode(x)
    target = feature_id.unsqueeze(1) if isinstance(feature_id, Tensor) else feature_id
    activation = (values * (indices == target)).sum(dim=-1)
    return activation.reshape(shape)


def apply_feature_intervention(
    residual: Tensor,
    sae: TopKSAE,
    feature_id: int | Tensor,
    mode: str,
    amount: float | Tensor,
    *,
    activation_scale: float = 1.0,
) -> Tensor:
    """Return a residual stream with SAE decoder-direction edits.

    The SAE operates on ``activation_scale * residual``. The edit is therefore
    calculated in the SAE's normalized coordinates and divided by that scale
    before it is added to the model's residual stream.
    """
    batched = isinstance(feature_id, Tensor)
    direction = sae.decoder_weight[feature_id]
    if batched:
        direction = direction.unsqueeze(1)
        amount = amount.unsqueeze(1)
    if mode == "clamp":
        normalized = residual.to(sae.decoder_weight.dtype) * activation_scale
        current = _feature_activation(sae, normalized, feature_id)
        delta = (amount - current).unsqueeze(-1) * direction
    elif mode == "additive":
        delta = amount.unsqueeze(-1) * direction if batched else amount * direction
    else:
        raise AssertionError(f"unexpected intervention mode: {mode}")
    return residual + (delta / activation_scale).to(residual.dtype)


@contextmanager
def feature_intervention_hook(
    layer: nn.Module,
    sae: TopKSAE,
    feature_id: int | Tensor,
    mode: str,
    amount: float | Tensor,
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
        self._baseline_cache: tuple[tuple, str, int] | None = None

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

        tokenizer = load_tokenizer(config["model_id"], tokenizer)
        model = load_causal_lm(
            config["model_id"],
            getattr(torch, config["model_dtype"]),
            device,
        )

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
        generation_args, prompt_tokens = self._prepare_generation(
            request.prompt, request
        )

        baseline_key = (
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
        )
        if self._baseline_cache is None or self._baseline_cache[0] != baseline_key:
            sampling_seed = torch.seed()
            torch.manual_seed(sampling_seed)
            baseline_ids = self.model.generate(**generation_args)
            baseline = self._decode_generation(baseline_ids[0], prompt_tokens)
            self._baseline_cache = baseline_key, baseline, sampling_seed
        else:
            _, baseline, sampling_seed = self._baseline_cache

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
            "baseline": baseline,
            "intervened": self._decode_generation(intervened_ids[0], prompt_tokens),
        }

    @torch.inference_mode()
    def generate_intervened(
        self, requests: list[InterventionRequest]
    ) -> list[str]:
        first = requests[0]
        generation_args, prompt_tokens = self._prepare_generation(
            [request.prompt for request in requests],
            first,
        )

        feature_ids = torch.tensor(
            [request.feature_id for request in requests],
            device=self.device,
        )
        amounts = torch.tensor(
            [request.amount for request in requests],
            device=self.device,
            dtype=self.sae.decoder_weight.dtype,
        )
        with feature_intervention_hook(
            self.layer,
            self.sae,
            feature_ids,
            first.mode,
            amounts,
            self.activation_scale,
        ):
            intervened_ids = self.model.generate(**generation_args)

        return [
            self._decode_generation(token_ids, prompt_tokens)
            for token_ids in intervened_ids
        ]

    def _prepare_generation(
        self,
        prompts: str | list[str],
        request: InterventionRequest,
    ) -> tuple[dict, int]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=isinstance(prompts, list),
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
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
        return generation_args, inputs["input_ids"].shape[1]

    def _decode_generation(self, token_ids: Tensor, prompt_tokens: int) -> str:
        return self.tokenizer.decode(
            token_ids[prompt_tokens:], skip_special_tokens=True
        )

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from .data import (
    RESIDUAL_FP16_SCALE,
    create_residual_cache,
    iter_context_batches,
    whole_contexts,
)
from .runtime import capture_layer_input
from .sae import (
    FIRING_THRESHOLD,
    RunningMetrics,
    TopKSAE,
    normalized_auxk_loss,
)

DOWNSTREAM_KL_TOKENS = 1_000_000
# Dimensions whose std exceeds this multiple of the median dimension's std are
# "massive activations" (attention-sink channels) and bypass the SAE unchanged.
PASSTHROUGH_STD_RATIO = 15.0


def default_aux_k(d_model: int) -> int:
    return 1 << round(math.log2(d_model / 2))


def sae_dim_mask(
    d_model: int, passthrough_dims: list[int], device: torch.device
) -> Tensor:
    keep = torch.ones(d_model, dtype=torch.bool, device=device)
    keep[passthrough_dims] = False
    return keep


def sae_input(residual: Tensor, passthrough_dims: list[int]) -> Tensor:
    # Drop pass-through dimensions from a (tokens, d_model) residual batch.
    if not passthrough_dims:
        return residual
    return residual[:, sae_dim_mask(residual.shape[-1], passthrough_dims, residual.device)]


@torch.inference_mode()
def reconstruct_residual(
    residual: Tensor,
    sae: TopKSAE,
    activation_scale: float,
    passthrough_dims: list[int],
) -> Tensor:
    # Replace the SAE-owned dimensions with their reconstruction; keep the rest.
    flat = residual.reshape(-1, residual.shape[-1])
    keep = sae_dim_mask(flat.shape[-1], passthrough_dims, flat.device)
    reconstruction = sae(flat[:, keep].float() * activation_scale)[0] / activation_scale
    output = flat.clone()
    output[:, keep] = reconstruction.to(flat.dtype)
    return output.reshape_as(residual)


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
    recording_tokens: int
    context_size: int
    layer_index: int
    width_multiplier: int
    k: int
    aux_k: int
    learning_rate: float
    gradient_clip: float | None
    model_batch_size: int
    sae_batch_size: int
    seed: int
    model_dtype: str
    cache_activations: bool
    passthrough_dims: list[int]
    normalization_tokens: int
    activation_scale: float
    subtract_pre_bias: bool
    aux_k_coef: float
    dead_window: int


@dataclass
class TrainingState:
    processed_tokens: int
    processed_batches: int
    last_fired: Tensor


def save_checkpoint(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: ExperimentConfig,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "sae": sae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "processed_tokens": state.processed_tokens,
            "processed_batches": state.processed_batches,
            "last_fired": state.last_fired.cpu(),
            "config": asdict(config),
        },
        temporary,
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def format_metrics(record: dict) -> str:
    dead = (
        "n/a"
        if record["dead_feature_pct"] is None
        else f"{record['dead_feature_pct']:.2f}%"
    )
    parts = []
    if record.get("downstream_kl") is not None:
        parts.append(f"downstream KL {record['downstream_kl']:.6g}")
    parts.extend(
        [
            f"EV {record['explained_variance']:.2%}",
            f"MSE {record['mse']:,.4f}",
        ]
    )
    if record.get("auxk_loss") is not None:
        parts.append(f"AuxK NMSE {record['auxk_loss']:,.4f}")
    parts.append(f"dead {dead}")
    return " | ".join(parts)


def format_metrics_line(record: dict) -> str:
    if record.get("split") == "validation":
        return f"Validation: {record['tokens']:,} tok | {format_metrics(record)}"
    return f"{'train':<10} {record['tokens']:>12,} tok | {format_metrics(record)}"


def load_evaluation(path: Path, training_tokens: int) -> dict | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        record = json.loads(line)
        if record["training_tokens"] == training_tokens:
            return {
                key: value
                for key, value in record.items()
                if key != "training_tokens"
            }
    return None


def capture_residuals(
    model: nn.Module, layer: nn.Module, input_ids: Tensor, device: torch.device
) -> Tensor:
    return capture_layer_input(model, layer, input_ids.to(device)).flatten(0, 1)


def iter_captured_residual_batches(
    model: nn.Module,
    layer: nn.Module,
    tokens: np.memmap,
    device: torch.device,
    context_size: int,
    model_batch_size: int,
    shuffle: bool,
    seed: int,
    skip_batches: int = 0,
) -> Iterator[Tensor]:
    batches = iter_context_batches(
        tokens,
        context_size,
        model_batch_size,
        shuffle=shuffle,
        seed=seed,
        skip_contexts=skip_batches * model_batch_size,
    )
    for input_ids in batches:
        yield capture_residuals(model, layer, input_ids, device)


@torch.inference_mode()
def capture_residual_cache(
    model: nn.Module,
    layer: nn.Module,
    train_tokens: np.memmap,
    validation_tokens: np.memmap,
    device: torch.device,
    context_size: int,
    model_batch_size: int,
    cache_paths: tuple[Path, Path, Path],
    metadata: dict,
) -> None:
    train_path, validation_path, metadata_path = cache_paths
    train_path.parent.mkdir(parents=True, exist_ok=True)
    float16_max = torch.finfo(torch.float16).max
    total_tokens = sum(
        whole_contexts(len(tokens), context_size)
        for tokens in (train_tokens, validation_tokens)
    )
    progress = tqdm(total=total_tokens, unit="tok", desc="Residual cache", dynamic_ncols=True)

    def capture_split(tokens: np.memmap, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        output = create_residual_cache(
            temporary, whole_contexts(len(tokens), context_size), metadata["d_model"]
        )
        written = 0
        for input_ids in iter_context_batches(
            tokens, context_size, model_batch_size, shuffle=False, seed=0
        ):
            # Scale down so outlier activations fit in fp16.
            stored = capture_residuals(model, layer, input_ids, device).float()
            stored.mul_(RESIDUAL_FP16_SCALE).clamp_(-float16_max, float16_max)
            output[written : written + len(stored)] = stored.cpu().numpy()
            written += len(stored)
            progress.update(len(stored))
        output.flush()
        del output
        os.replace(temporary, path)

    capture_split(train_tokens, train_path)
    capture_split(validation_tokens, validation_path)
    progress.close()
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(metadata_temporary, metadata_path)


def geometric_median(
    points: Tensor, *, max_iterations: int = 100, tolerance: float = 1e-5
) -> Tensor:
    estimate = points.mean(dim=0)
    for _ in range(max_iterations):
        distances = torch.linalg.vector_norm(points - estimate, dim=1)
        weights = distances.clamp_min(1e-7).reciprocal()
        updated = (points * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.linalg.vector_norm(updated - estimate) <= tolerance:
            return updated
        estimate = updated
    return estimate


def feature_density_histogram(fire_counts: Tensor, token_count: int) -> dict:
    nonzero_counts = fire_counts[fire_counts > 0].cpu().numpy().astype(np.float64)
    nonzero_density = nonzero_counts / token_count
    minimum_exponent = -math.ceil(math.log10(token_count))
    bin_edges = np.linspace(minimum_exponent, 0.0, -minimum_exponent * 10 + 1)
    bin_counts, bin_edges = np.histogram(np.log10(nonzero_density), bins=bin_edges)
    return {
        "total_features": fire_counts.numel(),
        "feature_density_log10_bin_edges": bin_edges.tolist(),
        "feature_density_bin_counts": bin_counts.tolist(),
    }


def optimize_residual_batch(
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    residual: Tensor,
    metrics: RunningMetrics,
    last_fired: Tensor,
    processed_tokens: int,
    sae_batch_size: int,
    gradient_clip: float | None,
    aux_k: int,
    aux_k_coef: float,
    dead_window: int,
) -> None:
    for start in range(0, len(residual), sae_batch_size):
        x = residual[start : start + sae_batch_size]
        reconstruction, indices, values, pre_activations = sae(x)
        token_position = processed_tokens + start + len(x)
        fired = indices[values > FIRING_THRESHOLD].unique()
        last_fired[fired] = token_position
        mse_loss = F.mse_loss(reconstruction, x)
        auxk_loss = None
        loss = mse_loss
        if aux_k_coef > 0 and token_position >= dead_window:
            dead_indices = torch.nonzero(
                last_fired < token_position - dead_window,
                as_tuple=True,
            )[0]
            if len(dead_indices) > 0:
                auxk_loss = normalized_auxk_loss(
                    sae,
                    pre_activations,
                    x - reconstruction,
                    dead_indices,
                    aux_k,
                )
                loss = loss + aux_k_coef * auxk_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.constrain_decoder_gradient()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), gradient_clip)
        optimizer.step()
        sae.normalize_decoder()

        metrics.update(
            x.detach(),
            reconstruction.detach(),
            indices.detach(),
            values.detach(),
            auxk_loss,
        )


@torch.inference_mode()
def calibrate_activations(
    train_batches,
    train_token_count: int,
    device: torch.device,
    normalization_tokens: int,
    subtract_pre_bias: bool,
    detect_passthrough: bool,
) -> tuple[list[int], float, Tensor | None]:
    # One pass: pick pass-through dims, then scale the rest so their mean
    # squared norm equals their width, and seed the pre-bias.
    target_tokens = min(normalization_tokens, train_token_count)
    tokens_seen = 0
    dim_sum = dim_sq_sum = first_batch = None
    progress = tqdm(
        total=target_tokens,
        unit="tok",
        desc="Calibrate activations",
        leave=False,
        dynamic_ncols=True,
    )
    for residual in train_batches(skip_batches=0):
        residual = residual[: target_tokens - tokens_seen]
        if first_batch is None:
            first_batch = residual.to(device=device, dtype=torch.float32)
        # Float64 avoids cancellation on dims with large constant offsets.
        residual = residual.to(device="cpu", dtype=torch.float64)
        batch_sum, batch_sq_sum = residual.sum(dim=0), residual.square().sum(dim=0)
        dim_sum = batch_sum if dim_sum is None else dim_sum + batch_sum
        dim_sq_sum = batch_sq_sum if dim_sq_sum is None else dim_sq_sum + batch_sq_sum
        tokens_seen += len(residual)
        progress.update(len(residual))
        if tokens_seen >= target_tokens:
            break
    progress.close()

    dim_std = (dim_sq_sum / tokens_seen - (dim_sum / tokens_seen).square()).sqrt()
    passthrough_dims = []
    if detect_passthrough:
        std_ratio = dim_std / dim_std.median()
        passthrough_dims = torch.nonzero(std_ratio > PASSTHROUGH_STD_RATIO).flatten().tolist()
        print(
            f"Pass-through dims: {passthrough_dims or 'none'}"
            + "".join(f" | {dim}: {std_ratio[dim]:.0f}x median std" for dim in passthrough_dims)
        )

    keep = sae_dim_mask(len(dim_std), passthrough_dims, torch.device("cpu"))
    mean_squared_norm = float(dim_sq_sum[keep].sum()) / tokens_seen
    scale = math.sqrt(int(keep.sum()) / mean_squared_norm)
    pre_bias = None
    if subtract_pre_bias:
        pre_bias = geometric_median(sae_input(first_batch, passthrough_dims)).mul_(scale)
    return passthrough_dims, scale, pre_bias


@torch.inference_mode()
def evaluate_downstream_kl(
    sae: TopKSAE,
    model: nn.Module,
    layer: nn.Module,
    validation_tokens: np.memmap,
    device: torch.device,
    context_size: int,
    activation_scale: float,
    passthrough_dims: list[int],
) -> dict:
    validation_subset = validation_tokens[:DOWNSTREAM_KL_TOKENS]
    context_kl_means = []
    progress = tqdm(
        total=whole_contexts(len(validation_subset), context_size),
        unit="tok",
        desc="Downstream KL",
        leave=False,
        disable=None,
    )

    def reconstruct_layer_input(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        reconstruction = reconstruct_residual(
            hidden, sae, activation_scale, passthrough_dims
        )
        if args:
            return (reconstruction, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": reconstruction}

    for input_ids in iter_context_batches(
        validation_subset,
        context_size,
        1,  # Keep full-vocabulary logits memory-bounded.
        shuffle=False,
        seed=0,
    ):
        model_kwargs = {"input_ids": input_ids.to(device), "use_cache": False}
        base_logits = model(**model_kwargs).logits[:, :-1].float()
        base_log_z = torch.logsumexp(base_logits, dim=-1)

        handle = layer.register_forward_pre_hook(
            reconstruct_layer_input, with_kwargs=True
        )
        try:
            sae_logits = model(**model_kwargs).logits[:, :-1].float()
        finally:
            handle.remove()
        sae_log_z = torch.logsumexp(sae_logits, dim=-1)

        # KL(base || sae), reusing the logit tensors to avoid another vocab-sized buffer.
        sae_logits.neg_().add_(base_logits)
        base_logits.sub_(base_log_z.unsqueeze(-1)).exp_()
        token_kl = sae_logits.mul_(base_logits).sum(dim=-1)
        token_kl.add_(sae_log_z).sub_(base_log_z)

        context_kl_means.append(token_kl.mean().item())
        progress.update(input_ids.numel())

    progress.close()
    # Every context has the same number of predictions, so this is the token mean.
    mean_kl = float(np.mean(context_kl_means))
    standard_error = float(
        np.std(context_kl_means, ddof=1) / math.sqrt(len(context_kl_means))
    )
    margin = 1.96 * standard_error
    return {
        "downstream_kl": mean_kl,
        "downstream_kl_ci": [max(0.0, mean_kl - margin), mean_kl + margin],
    }


def train_sae(
    sae: TopKSAE,
    train_batches,
    validation_batches,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: Path,
    resume: bool,
    log_every: int,
    checkpoint_every: int,
    validate_every: int,
    downstream_kl_evaluator: Callable[[TopKSAE, ExperimentConfig], dict],
) -> dict:
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    evaluation_metrics_path = output_dir / "evaluation_metrics.jsonl"
    if resume:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint["config"] != asdict(config):
            raise ValueError("cannot resume with a different experiment configuration")
        sae.load_state_dict(checkpoint["sae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = TrainingState(
            processed_tokens=int(checkpoint["processed_tokens"]),
            processed_batches=int(checkpoint["processed_batches"]),
            last_fired=checkpoint["last_fired"].to(device),
        )
        print(f"resumed at {state.processed_tokens:,} training tokens")
    else:
        metrics_path.write_text("")
        evaluation_metrics_path.write_text("")
        state = TrainingState(
            processed_tokens=0,
            processed_batches=0,
            last_fired=torch.full(
                (sae.d_sae,), -1, dtype=torch.int64, device=device
            ),
        )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    latest_evaluation = None
    evaluation_seconds = 0.0
    train_tokens = whole_contexts(config.train_tokens, config.context_size)

    def evaluate() -> dict:
        nonlocal evaluation_seconds
        evaluation_start = time.monotonic()
        evaluation = evaluate_sae(sae, validation_batches, device, config)
        evaluation.update(downstream_kl_evaluator(sae, config))
        evaluation_seconds += time.monotonic() - evaluation_start
        evaluation_record = {
            **evaluation,
            "training_tokens": state.processed_tokens,
        }
        append_jsonl(evaluation_metrics_path, evaluation_record)
        tqdm.write(format_metrics_line(evaluation_record))
        return evaluation

    if resume:
        latest_evaluation = load_evaluation(
            evaluation_metrics_path, state.processed_tokens
        )

    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    next_log = ((state.processed_tokens // log_every) + 1) * log_every
    next_checkpoint = (
        (state.processed_tokens // checkpoint_every) + 1
    ) * checkpoint_every
    next_validation = (
        (state.processed_tokens // validate_every) + 1
    ) * validate_every
    progress = tqdm(
        total=train_tokens,
        initial=state.processed_tokens,
        unit="tok",
        desc="Train",
        dynamic_ncols=True, disable=None,
    )
    metric_status = tqdm(
        desc="Metrics", bar_format="{desc}", dynamic_ncols=True, disable=progress.disable
    )
    start_time = time.monotonic()
    start_tokens = state.processed_tokens
    last_evaluated_training_tokens = (
        state.processed_tokens if latest_evaluation is not None else None
    )
    for residual in train_batches(skip_batches=state.processed_batches):
        batch_index = state.processed_batches
        state.processed_batches += 1
        residual = residual.to(device=device, dtype=torch.float32)
        residual = sae_input(residual, config.passthrough_dims)
        residual.mul_(config.activation_scale)
        batch_tokens = len(residual)
        torch.manual_seed(config.seed + batch_index)
        permutation = torch.randperm(len(residual), device=residual.device)
        residual = residual[permutation]

        optimize_residual_batch(
            sae,
            optimizer,
            residual,
            metrics,
            state.last_fired,
            state.processed_tokens,
            config.sae_batch_size,
            config.gradient_clip,
            config.aux_k,
            config.aux_k_coef,
            config.dead_window,
        )
        state.processed_tokens += batch_tokens
        progress.update(batch_tokens)

        if (
            state.processed_tokens >= next_log
            or state.processed_tokens == train_tokens
        ):
            if state.processed_tokens >= config.dead_window:
                dead_features = (
                    state.last_fired
                    < state.processed_tokens - config.dead_window
                )
                dead_feature_pct = 100.0 * dead_features.float().mean().item()
            else:
                dead_feature_pct = None
            record = {
                "tokens": state.processed_tokens,
                **metrics.compute(),
                "dead_feature_pct": dead_feature_pct,
                "tokens_per_second": (state.processed_tokens - start_tokens)
                / (time.monotonic() - start_time - evaluation_seconds),
            }
            append_jsonl(metrics_path, record)
            if progress.disable:
                print(format_metrics_line(record))
            else:
                metric_status.set_description_str(
                    f"Metrics\t{format_metrics(record)}", refresh=True
                )
            metrics.reset()
            while next_log <= state.processed_tokens:
                next_log += log_every

        if state.processed_tokens >= next_checkpoint:
            save_checkpoint(checkpoint_path, sae, optimizer, state, config)
            while next_checkpoint <= state.processed_tokens:
                next_checkpoint += checkpoint_every

        if state.processed_tokens >= next_validation:
            latest_evaluation = evaluate()
            last_evaluated_training_tokens = state.processed_tokens
            while next_validation <= state.processed_tokens:
                next_validation += validate_every

    metric_status.close()
    progress.close()
    save_checkpoint(checkpoint_path, sae, optimizer, state, config)
    if last_evaluated_training_tokens != state.processed_tokens:
        latest_evaluation = evaluate()
    torch.save(
        {"sae": sae.state_dict()},
        output_dir / "sae_final.pt",
    )
    return latest_evaluation


@torch.inference_mode()
def evaluate_sae(
    sae: TopKSAE,
    validation_batches,
    device: torch.device,
    config: ExperimentConfig,
) -> dict:
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    progress = tqdm(
        total=whole_contexts(config.validation_tokens, config.context_size),
        unit="tok",
        desc="Validate",
        leave=False,
        disable=None,
    )
    evaluated_tokens = 0
    for residual in validation_batches(skip_batches=0):
        residual = residual.to(device=device, dtype=torch.float32)
        residual = sae_input(residual, config.passthrough_dims)
        residual.mul_(config.activation_scale)
        batch_tokens = len(residual)
        evaluated_tokens += batch_tokens
        for start in range(0, len(residual), config.sae_batch_size):
            x = residual[start : start + config.sae_batch_size]
            reconstruction, indices, values, _ = sae(x)
            metrics.update(x, reconstruction, indices, values)
        progress.update(batch_tokens)
    progress.close()

    fire_counts = metrics.feature_fire_counts
    result = {
        "split": "validation",
        "tokens": evaluated_tokens,
        **metrics.compute(),
        "dead_feature_pct": 100.0 * (fire_counts == 0).float().mean().item(),
        "active_features": int((fire_counts > 0).sum().item()),
        **feature_density_histogram(fire_counts, evaluated_tokens),
    }
    return result

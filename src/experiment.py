from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from .data import RESIDUAL_DTYPE, iter_context_batches, iter_residual_batches
from .sae import (
    FIRING_THRESHOLD,
    RunningMetrics,
    TopKSAE,
    normalized_auxk_loss,
)


def default_aux_k(d_model: int) -> int:
    return 1 << round(math.log2(d_model / 2))


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
    context_size: int
    layer_index: int
    layer_path: str
    residual_location: str
    d_model: int
    d_sae: int
    width_multiplier: int
    k: int
    aux_k: int
    learning_rate: float
    model_batch_size: int
    sae_batch_size: int
    seed: int
    device: str
    model_dtype: str
    normalization_tokens: int = 0
    activation_scale: float = 1.0
    subtract_pre_bias: bool = True
    aux_k_coef: float = 1 / 32
    dead_window: int = 10_000_000


@dataclass
class TrainingState:
    processed_tokens: int
    processed_batches: int
    last_fired: Tensor


def find_transformer_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.ModuleList)
            and name.split(".")[-1] == "layers"
            and len(module) > 1
        ):
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("could not locate the transformer's ModuleList named 'layers'")
    return max(candidates, key=lambda item: len(item[1]))


class _ActivationCaptured(Exception):
    pass


class ResidualStreamCapture:
    def __init__(self, layer: nn.Module):
        self.activation: Tensor | None = None

        def capture(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self.activation = hidden.detach()
            raise _ActivationCaptured

        self.handle = layer.register_forward_pre_hook(capture, with_kwargs=True)

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, input_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        self.activation = None
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _ActivationCaptured:
            pass
        if self.activation is None:
            raise RuntimeError("the residual-stream hook did not run")
        return self.activation

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> ResidualStreamCapture:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


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
    parts = [
        f"EV {record['explained_variance']:.2%}",
        f"MSE {record['mse']:,.4f}",
    ]
    if "auxk_loss" in record:
        parts.append(f"AuxK NMSE {record['auxk_loss']:,.4f}")
    parts.append(f"dead {dead}")
    return " | ".join(parts)


def format_metrics_line(record: dict) -> str:
    return f"{record['split']:<10} {record['tokens']:>12,} tok | {format_metrics(record)}"


def load_checkpoint_evaluation(path: Path, training_tokens: int) -> dict | None:
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


def capture_residual_batch(
    model: nn.Module,
    capture: ResidualStreamCapture,
    input_ids: Tensor,
    attention_mask: Tensor,
    batch_tokens: int,
    device: torch.device,
) -> Tensor:
    full_batch = batch_tokens == input_ids.numel()
    input_ids = input_ids.to(device, non_blocking=True)
    if full_batch:
        return capture(model, input_ids, None).flatten(0, 1)

    attention_mask = attention_mask.to(device, non_blocking=True)
    valid_mask = attention_mask.bool()
    return capture(model, input_ids, attention_mask)[valid_mask]


@torch.inference_mode()
def capture_residual_cache(
    model: nn.Module,
    capture: ResidualStreamCapture,
    train_tokens: np.memmap,
    validation_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    cache_paths: tuple[Path, Path, Path],
    metadata: dict,
) -> None:
    train_path, validation_path, metadata_path = cache_paths
    train_path.parent.mkdir(parents=True, exist_ok=True)
    d_model = int(metadata["d_model"])
    total_tokens = len(train_tokens) + len(validation_tokens)
    progress = tqdm(
        total=total_tokens,
        unit="tok",
        desc="Residual cache",
        dynamic_ncols=True,
    )

    def capture_split(tokens: np.memmap, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        output = np.memmap(
            temporary,
            mode="w+",
            dtype=RESIDUAL_DTYPE,
            shape=(len(tokens), d_model),
        )
        written = 0
        batches = iter_context_batches(
            tokens,
            args.context_size,
            args.model_batch_size,
            pad_token_id,
            shuffle=False,
            seed=0,
        )
        for input_ids, attention_mask, _context_ids in batches:
            batch_tokens = int(attention_mask.sum())
            residual = capture_residual_batch(
                model, capture, input_ids, attention_mask, batch_tokens, device
            )
            stored = residual.float().cpu().numpy()
            output[written : written + batch_tokens] = stored
            written += batch_tokens
            progress.update(batch_tokens)
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
        reconstruction, indices, values, pre_activations = (
            sae.forward_with_pre_activations(x)
        )
        token_position = processed_tokens + start + len(x)
        fired = indices[values > FIRING_THRESHOLD].unique()
        last_fired[fired] = token_position
        mse_loss = F.mse_loss(reconstruction, x)
        auxk_loss = None
        loss = mse_loss
        if aux_k_coef > 0:
            dead_mask = last_fired < token_position - dead_window
            if dead_mask.any():
                auxk_loss = normalized_auxk_loss(
                    sae,
                    pre_activations,
                    x - reconstruction,
                    dead_mask,
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
def estimate_activation_normalization(
    train_residuals: np.memmap,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, Tensor | None]:
    target_tokens = min(args.normalization_tokens, len(train_residuals))
    batches = iter_residual_batches(
        train_residuals,
        args.context_size * args.model_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    tokens_seen = 0
    squared_norm_sum = 0.0
    pre_bias = None
    pre_bias_sample_tokens = 0
    progress = tqdm(
        total=target_tokens,
        unit="tok",
        desc="Calibrate activation scale",
        leave=False,
        dynamic_ncols=True,
    )
    for residual in batches:
        residual = residual.to(device=device, dtype=torch.float32)
        take = min(len(residual), target_tokens - tokens_seen)
        calibration_residual = residual[:take]
        if args.subtract_pre_bias and pre_bias is None:
            pre_bias = geometric_median(calibration_residual)
            pre_bias_sample_tokens = take
        squared_norm_sum += calibration_residual.square().sum().item()
        tokens_seen += take
        progress.update(take)
        if tokens_seen >= target_tokens:
            break
    progress.close()

    mean_squared_norm = squared_norm_sum / tokens_seen
    scale = math.sqrt(train_residuals.shape[1] / mean_squared_norm)
    print(
        f"activation normalization: {tokens_seen:,} calibration tokens, "
        f"E[||x||^2]={mean_squared_norm:,.4g}, scale={scale:.8g}"
    )
    if pre_bias is not None:
        pre_bias.mul_(scale)
        print(
            f"pre-bias initialization: geometric median of "
            f"{pre_bias_sample_tokens:,} scaled calibration activations"
        )
    return scale, pre_bias


def train_sae(
    sae: TopKSAE,
    train_residuals: np.memmap,
    validation_residuals: np.memmap,
    device: torch.device,
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> tuple[int, dict | None]:
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    metrics_path = args.output_dir / "train_metrics.jsonl"
    checkpoint_metrics_path = args.output_dir / "checkpoint_metrics.jsonl"
    if args.resume:
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
        checkpoint_metrics_path.write_text("")
        state = TrainingState(
            processed_tokens=0,
            processed_batches=0,
            last_fired=torch.full(
                (sae.d_sae,), -1, dtype=torch.int64, device=device
            ),
        )
    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    latest_evaluation = None
    evaluation_seconds = 0.0

    def evaluate_checkpoint() -> dict:
        nonlocal evaluation_seconds
        evaluation_start = time.monotonic()
        evaluation = evaluate_sae(
            sae,
            validation_residuals,
            device,
            config,
        )
        evaluation_seconds += time.monotonic() - evaluation_start
        checkpoint_record = {
            **evaluation,
            "training_tokens": state.processed_tokens,
        }
        append_jsonl(checkpoint_metrics_path, checkpoint_record)
        tqdm.write(format_metrics_line(checkpoint_record))
        return evaluation

    if args.resume:
        latest_evaluation = load_checkpoint_evaluation(
            checkpoint_metrics_path, state.processed_tokens
        )
        if latest_evaluation is None:
            latest_evaluation = evaluate_checkpoint()

    residual_batch_size = config.context_size * config.model_batch_size
    batches = iter_residual_batches(
        train_residuals,
        residual_batch_size,
        shuffle=True,
        seed=config.seed,
        skip_batches=state.processed_batches,
    )
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    next_log = ((state.processed_tokens // args.log_every) + 1) * args.log_every
    next_checkpoint = (
        (state.processed_tokens // args.checkpoint_every) + 1
    ) * args.checkpoint_every
    progress = tqdm(
        total=len(train_residuals),
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
    evaluation_seconds = 0.0
    for residual in batches:
        residual = residual.to(device=device, dtype=torch.float32)
        residual.mul_(config.activation_scale)
        batch_tokens = len(residual)
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
            args.gradient_clip,
            config.aux_k,
            config.aux_k_coef,
            config.dead_window,
        )
        state.processed_tokens += batch_tokens
        state.processed_batches += 1
        progress.update(batch_tokens)

        if (
            state.processed_tokens >= next_log
            or state.processed_tokens == len(train_residuals)
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
                "split": "train",
                "tokens": state.processed_tokens,
                **metrics.compute(),
                "dead_feature_pct": dead_feature_pct,
                "learning_rate": config.learning_rate,
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
                next_log += args.log_every

        if (
            state.processed_tokens >= next_checkpoint
            or state.processed_tokens == len(train_residuals)
        ):
            save_checkpoint(checkpoint_path, sae, optimizer, state, config)
            latest_evaluation = evaluate_checkpoint()
            while next_checkpoint <= state.processed_tokens:
                next_checkpoint += args.checkpoint_every

    metric_status.close()
    progress.close()
    torch.save(
        {"sae": sae.state_dict(), "config": asdict(config)},
        args.output_dir / "sae_final.pt",
    )
    return state.processed_tokens, latest_evaluation


@torch.inference_mode()
def evaluate_sae(
    sae: TopKSAE,
    validation_residuals: np.memmap,
    device: torch.device,
    config: ExperimentConfig,
) -> dict:
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    batches = iter_residual_batches(
        validation_residuals,
        config.context_size * config.model_batch_size,
        shuffle=False,
        seed=config.seed,
    )
    progress = tqdm(
        total=len(validation_residuals), unit="tok", desc="Validate", leave=False, disable=None
    )
    for residual in batches:
        residual = residual.to(device=device, dtype=torch.float32)
        residual.mul_(config.activation_scale)
        batch_tokens = len(residual)
        for start in range(0, len(residual), config.sae_batch_size):
            x = residual[start : start + config.sae_batch_size]
            reconstruction, indices, values = sae(x)
            metrics.update(x, reconstruction, indices, values)
        progress.update(batch_tokens)
    progress.close()

    fire_counts = metrics.feature_fire_counts
    result = {
        "split": "validation",
        "tokens": len(validation_residuals),
        **metrics.compute(),
        "dead_feature_pct": 100.0 * (fire_counts == 0).float().mean().item(),
        "active_features": int((fire_counts > 0).sum().item()),
        **feature_density_histogram(fire_counts, len(validation_residuals)),
    }
    return result

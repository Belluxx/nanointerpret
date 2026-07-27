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

from .data import iter_context_batches
from .sae import (
    FIRING_THRESHOLD,
    RunningMetrics,
    TopKSAE,
    normalized_auxk_loss,
)


def default_aux_k(d_model: int) -> int:
    """Return a power of two close to half the residual-stream width."""
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    return 1 << round(math.log2(max(1.0, d_model / 2)))


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
    learning_rate: float
    model_batch_size: int
    sae_batch_size: int
    seed: int
    device: str
    model_dtype: str
    normalization_tokens: int = 0
    activation_scale: float = 1.0
    aux_k: int | None = None
    aux_k_coef: float = 1 / 32
    dead_window: int = 10_000_000

    def __post_init__(self) -> None:
        aux_k = default_aux_k(self.d_model) if self.aux_k is None else self.aux_k
        if not 0 < aux_k <= self.d_sae:
            raise ValueError(f"aux_k must be in [1, {self.d_sae}], got {aux_k}")
        if self.aux_k_coef < 0:
            raise ValueError("aux_k_coef must be non-negative")
        if self.dead_window <= 0:
            raise ValueError("dead_window must be positive")
        object.__setattr__(self, "aux_k", aux_k)


@dataclass
class TrainingState:
    processed_tokens: int
    processed_contexts: int
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
    """Capture a layer's input residual and stop the rest of the model forward."""

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
    processed_tokens: int,
    processed_contexts: int,
    last_fired: Tensor,
    config: ExperimentConfig,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "sae": sae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "processed_tokens": processed_tokens,
            "processed_contexts": processed_contexts,
            "last_fired": last_fired.cpu(),
            "config": asdict(config),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_training_state(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
) -> TrainingState:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_resume_config(checkpoint.get("config"), config)
    sae.load_state_dict(checkpoint["sae"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    processed_tokens = int(checkpoint["processed_tokens"])
    processed_contexts = int(
        checkpoint.get(
            "processed_contexts",
            math.ceil(processed_tokens / config.context_size),
        )
    )
    return TrainingState(
        processed_tokens=processed_tokens,
        processed_contexts=processed_contexts,
        last_fired=checkpoint["last_fired"].to(device),
    )


def initialize_training_state(
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    metrics_path: Path,
    checkpoint_metrics_path: Path,
) -> TrainingState:
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume: {checkpoint_path} does not exist")
        state = load_training_state(
            checkpoint_path,
            sae,
            optimizer,
            config,
            device,
        )
        print(f"resumed at {state.processed_tokens:,} training tokens")
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists; "
                "pass --resume or choose another --output-dir"
            )
        metrics_path.write_text("")
        checkpoint_metrics_path.write_text("")
        state = TrainingState(
            processed_tokens=0,
            processed_contexts=0,
            last_fired=torch.full(
                (sae.d_sae,),
                -1,
                dtype=torch.int64,
                device=device,
            ),
        )

    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    return state


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def format_metrics(record: dict) -> str:
    """Format the useful metrics for terminal output."""
    dead = (
        "n/a"
        if record["dead_feature_pct"] is None
        else f"{record['dead_feature_pct']:.2f}%"
    )
    parts = [
        f"EV {record['explained_variance']:.2%}",
        f"recon {record['reconstruction_loss']:,.4f}",
    ]
    if "auxk_loss" in record:
        parts.append(f"AuxK {record['auxk_loss']:,.4f}")
    parts.extend(
        (f"NMSE {record['normalized_mse']:.4f}", f"dead {dead}")
    )
    return " | ".join(parts)


def format_metrics_line(record: dict) -> str:
    return f"{record['split']:<10} {record['tokens']:>12,} tok | {format_metrics(record)}"


def load_checkpoint_evaluation(path: Path, training_tokens: int) -> dict | None:
    """Return the latest validation for a training checkpoint, if present."""
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("training_tokens") == training_tokens
            and "feature_density_bin_counts" in record
        ):
            return {
                key: value
                for key, value in record.items()
                if key != "training_tokens"
            }
    return None


def validate_resume_config(checkpoint_config: object, config: ExperimentConfig) -> None:
    current_config = asdict(config)
    if not isinstance(checkpoint_config, dict):
        raise ValueError("cannot resume: checkpoint has no valid experiment configuration")

    fields = sorted(checkpoint_config.keys() | current_config.keys())
    mismatches = [
        field
        for field in fields
        if checkpoint_config.get(field) != current_config.get(field)
        or (field in checkpoint_config) != (field in current_config)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: checkpoint={checkpoint_config.get(field)!r}, "
            f"current={current_config.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"cannot resume with a different experiment configuration ({details})")


def capture_residual_batch(
    model: nn.Module,
    capture: ResidualStreamCapture,
    input_ids: Tensor,
    attention_mask: Tensor,
    batch_tokens: int,
    device: torch.device,
    activation_scale: float = 1.0,
) -> Tensor:
    """Capture and globally scale packed residuals."""
    full_batch = batch_tokens == input_ids.numel()
    input_ids = input_ids.to(device, non_blocking=True)
    if full_batch:
        residual = capture(model, input_ids, None).flatten(0, 1).float()
        residual.mul_(activation_scale)
        return residual

    attention_mask = attention_mask.to(device, non_blocking=True)
    valid_mask = attention_mask.bool()
    residual = capture(model, input_ids, attention_mask)[valid_mask].float()
    residual.mul_(activation_scale)
    return residual


def feature_density_histogram(fire_counts: Tensor, token_count: int) -> dict:
    """Build fixed log10-density bins suitable for comparison across validations."""
    nonzero_counts = fire_counts[fire_counts > 0].cpu().numpy().astype(np.float64)
    nonzero_density = nonzero_counts / token_count
    minimum_exponent = -max(1, math.ceil(math.log10(token_count)))
    bin_edges = np.linspace(minimum_exponent, 0.0, -minimum_exponent * 10 + 1)
    bin_counts, bin_edges = np.histogram(np.log10(nonzero_density), bins=bin_edges)
    return {
        "total_features": fire_counts.numel(),
        "feature_density_log10_bin_edges": bin_edges.tolist(),
        "feature_density_bin_counts": bin_counts.tolist(),
    }


@torch.no_grad()
def update_last_fired(
    last_fired: Tensor,
    indices: Tensor,
    values: Tensor,
    token_position: int,
) -> None:
    """Record primary TopK firings before dead latents are selected for AuxK."""
    fired = indices[values > FIRING_THRESHOLD].unique()
    last_fired[fired] = token_position


def optimize_residual_batch(
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    residual: Tensor,
    metrics: RunningMetrics,
    last_fired: Tensor,
    processed_tokens: int,
    sae_batch_size: int,
    gradient_clip: float,
    aux_k: int,
    aux_k_coef: float,
    dead_window: int,
) -> None:
    """Optimize the SAE over one captured model batch."""
    for start in range(0, len(residual), sae_batch_size):
        x = residual[start : start + sae_batch_size]
        reconstruction, indices, values, pre_activations = (
            sae.forward_with_pre_activations(x)
        )
        token_position = processed_tokens + start + len(x)
        update_last_fired(last_fired, indices, values, token_position)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        auxk_loss = None
        loss = reconstruction_loss
        if aux_k_coef > 0:
            dead_mask = last_fired < token_position - dead_window
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
        if gradient_clip > 0:
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


def build_training_record(
    metrics: RunningMetrics,
    last_fired: Tensor,
    processed_tokens: int,
    dead_window: int,
    learning_rate: float,
    start_tokens: int,
    elapsed_seconds: float,
) -> dict:
    dead_feature_pct = (
        100.0
        * (last_fired < processed_tokens - dead_window).float().mean().item()
        if processed_tokens >= dead_window
        else None
    )
    return {
        "split": "train",
        "tokens": processed_tokens,
        **metrics.compute(),
        "dead_feature_pct": dead_feature_pct,
        "learning_rate": learning_rate,
        "tokens_per_second": (processed_tokens - start_tokens)
        / max(elapsed_seconds, 1e-9),
    }


@torch.inference_mode()
def estimate_activation_normalization(
    model: nn.Module,
    capture: ResidualStreamCapture,
    train_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    d_model: int,
) -> float:
    """Estimate one global activation scale from a training calibration sample."""
    target_tokens = min(args.normalization_tokens, len(train_tokens))
    batches = iter_context_batches(
        train_tokens,
        args.context_size,
        args.model_batch_size,
        pad_token_id,
        shuffle=True,
        seed=args.seed,
    )
    tokens_seen = 0
    squared_norm_sum = 0.0
    progress = tqdm(
        total=target_tokens,
        unit="tok",
        desc="Calibrate activation scale",
        leave=False,
        dynamic_ncols=True,
    )
    for input_ids, attention_mask, _context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual = capture_residual_batch(
            model, capture, input_ids, attention_mask, batch_tokens, device
        )
        take = min(len(residual), target_tokens - tokens_seen)
        calibration_residual = residual[:take]
        squared_norm_sum += calibration_residual.square().sum().item()
        tokens_seen += take
        progress.update(take)
        if tokens_seen >= target_tokens:
            break
    progress.close()

    mean_squared_norm = squared_norm_sum / max(tokens_seen, 1)
    if not math.isfinite(mean_squared_norm) or mean_squared_norm <= 0:
        raise RuntimeError(
            f"cannot normalize activations with E[||x||^2]={mean_squared_norm}"
        )
    scale = math.sqrt(d_model / mean_squared_norm)
    print(
        f"activation normalization: {tokens_seen:,} calibration tokens, "
        f"E[||x||^2]={mean_squared_norm:,.4g}, scale={scale:.8g}"
    )
    return scale


def train_sae(
    model: nn.Module,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    train_tokens: np.memmap,
    validation_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> tuple[int, dict | None]:
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    metrics_path = args.output_dir / "train_metrics.jsonl"
    checkpoint_metrics_path = args.output_dir / "checkpoint_metrics.jsonl"
    state = initialize_training_state(
        sae,
        optimizer,
        config,
        device,
        args,
        checkpoint_path,
        metrics_path,
        checkpoint_metrics_path,
    )
    latest_evaluation = None
    evaluation_seconds = 0.0

    def evaluate_checkpoint() -> dict:
        nonlocal evaluation_seconds
        evaluation_start = time.monotonic()
        evaluation = evaluate_sae(
            model,
            capture,
            sae,
            validation_tokens,
            pad_token_id,
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

    batches = iter_context_batches(
        train_tokens,
        config.context_size,
        config.model_batch_size,
        pad_token_id,
        shuffle=True,
        seed=config.seed,
        skip_contexts=state.processed_contexts,
    )
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    next_log = ((state.processed_tokens // args.log_every) + 1) * args.log_every
    next_checkpoint = (
        (state.processed_tokens // args.checkpoint_every) + 1
    ) * args.checkpoint_every
    progress = tqdm(
        total=len(train_tokens),
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
    for input_ids, attention_mask, context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual = capture_residual_batch(
            model,
            capture,
            input_ids,
            attention_mask,
            batch_tokens,
            device,
            config.activation_scale,
        )
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
            int(config.aux_k),
            config.aux_k_coef,
            config.dead_window,
        )
        state.processed_tokens += batch_tokens
        state.processed_contexts += len(context_ids)
        progress.update(batch_tokens)

        if (
            state.processed_tokens >= next_log
            or state.processed_tokens == len(train_tokens)
        ):
            record = build_training_record(
                metrics,
                state.last_fired,
                state.processed_tokens,
                config.dead_window,
                config.learning_rate,
                start_tokens,
                time.monotonic() - start_time - evaluation_seconds,
            )
            append_jsonl(metrics_path, record)
            if progress.disable:
                # Redirected output cannot redraw a line, so emit readable log records.
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
            or state.processed_tokens == len(train_tokens)
        ):
            save_checkpoint(
                checkpoint_path,
                sae,
                optimizer,
                state.processed_tokens,
                state.processed_contexts,
                state.last_fired,
                config,
            )
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
    model: nn.Module,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    validation_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    config: ExperimentConfig,
) -> dict:
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    batches = iter_context_batches(
        validation_tokens,
        config.context_size,
        config.model_batch_size,
        pad_token_id,
        shuffle=False,
        seed=config.seed,
    )
    progress = tqdm(
        total=len(validation_tokens), unit="tok", desc="Validate", leave=False, disable=None
    )
    for input_ids, attention_mask, _context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual = capture_residual_batch(
            model,
            capture,
            input_ids,
            attention_mask,
            batch_tokens,
            device,
            config.activation_scale,
        )
        for start in range(0, len(residual), config.sae_batch_size):
            x = residual[start : start + config.sae_batch_size]
            reconstruction, indices, values = sae(x)
            metrics.update(x, reconstruction, indices, values)
        progress.update(batch_tokens)
    progress.close()

    fire_counts = metrics.feature_fire_counts
    result = {
        "split": "validation",
        "tokens": len(validation_tokens),
        **metrics.compute(),
        "dead_feature_pct": 100.0 * (fire_counts == 0).float().mean().item(),
        "active_features": int((fire_counts > 0).sum().item()),
        **feature_density_histogram(fire_counts, len(validation_tokens)),
    }
    return result

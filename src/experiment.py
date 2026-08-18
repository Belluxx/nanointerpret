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
    encode_int8_residuals,
    iter_context_batches,
)
from .sae import (
    FIRING_THRESHOLD,
    RunningMetrics,
    TopKSAE,
    normalized_auxk_loss,
)

ResidualBatchFactory = Callable[..., Iterator[Tensor]]


def default_aux_k(d_model: int) -> int:
    return 1 << round(math.log2(d_model / 2))


def raw_l2_activation_mask(residual: Tensor, max_activation_l2: float | None) -> Tensor:
    # Return which raw residual vectors are eligible for SAE consumption.
    if max_activation_l2 is None:
        return torch.ones(residual.shape[:-1], dtype=torch.bool, device=residual.device)
    return torch.linalg.vector_norm(residual.float(), dim=-1) <= max_activation_l2


def filter_raw_l2_activations(
    residual: Tensor, max_activation_l2: float | None
) -> Tensor:
    # Drop raw residual vectors above the L2 threshold
    if max_activation_l2 is None:
        return residual
    return residual[raw_l2_activation_mask(residual, max_activation_l2)]


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
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
    residual_cache_format: str | None
    max_activation_l2: float | None = None
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
    if record.get("auxk_loss") is not None:
        parts.append(f"AuxK NMSE {record['auxk_loss']:,.4f}")
    parts.append(f"dead {dead}")
    return " | ".join(parts)


def format_metrics_line(record: dict) -> str:
    if record.get("split") == "validation":
        return f"Validation: {record['tokens']:,} tok | {format_metrics(record)}"
    return f"{'train':<10} {record['tokens']:>12,} tok | {format_metrics(record)}"


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


def iter_captured_residual_batches(
    model: nn.Module,
    capture: ResidualStreamCapture,
    tokens: np.memmap,
    pad_token_id: int,
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
        pad_token_id,
        shuffle=shuffle,
        seed=seed,
        skip_contexts=skip_batches * model_batch_size,
    )
    for input_ids, attention_mask in batches:
        batch_tokens = int(attention_mask.sum())
        yield capture_residual_batch(
            model, capture, input_ids, attention_mask, batch_tokens, device
        )


@torch.inference_mode()
def capture_residual_cache(
    model: nn.Module,
    capture: ResidualStreamCapture,
    train_tokens: np.memmap,
    validation_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    context_size: int,
    model_batch_size: int,
    cache_paths: tuple[Path, Path, Path],
    metadata: dict,
) -> None:
    train_path, validation_path, metadata_path = cache_paths
    train_path.parent.mkdir(parents=True, exist_ok=True)
    d_model = int(metadata["d_model"])
    cache_format = metadata["cache_format"]
    float16_max = torch.finfo(torch.float16).max
    total_tokens = len(train_tokens) + len(validation_tokens)
    progress = tqdm(
        total=total_tokens,
        unit="tok",
        desc="Residual cache",
        dynamic_ncols=True,
    )

    def capture_split(tokens: np.memmap, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        output = create_residual_cache(
            temporary, len(tokens), d_model, cache_format
        )
        written = 0
        batches = iter_context_batches(
            tokens,
            context_size,
            model_batch_size,
            pad_token_id,
            shuffle=False,
            seed=0,
        )
        for input_ids, attention_mask in batches:
            batch_tokens = int(attention_mask.sum())
            residual = capture_residual_batch(
                model, capture, input_ids, attention_mask, batch_tokens, device
            )
            residual = residual.float()
            output_slice = slice(written, written + batch_tokens)
            if cache_format == "fp16":
                stored = residual.mul(RESIDUAL_FP16_SCALE)
                stored.clamp_(-float16_max, float16_max)
                output[output_slice] = stored.cpu().numpy()
            else:
                codes, scales = encode_int8_residuals(residual)
                output["codes"][output_slice] = codes.cpu().numpy()
                output["scales"][output_slice] = scales.cpu().numpy()
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
def estimate_activation_normalization(
    train_batches: ResidualBatchFactory,
    train_token_count: int,
    d_model: int,
    device: torch.device,
    normalization_tokens: int,
    subtract_pre_bias: bool,
    max_activation_l2: float | None,
) -> tuple[float, Tensor | None]:
    target_tokens = min(normalization_tokens, train_token_count)
    tokens_seen = 0
    squared_norm_sum = 0.0
    pre_bias = None
    progress = tqdm(
        total=target_tokens,
        unit="tok",
        desc="Calibrate activation scale",
        leave=False,
        dynamic_ncols=True,
    )
    for residual in train_batches(skip_batches=0):
        residual = filter_raw_l2_activations(residual, max_activation_l2)
        if len(residual) == 0:
            continue
        residual = residual.to(device=device, dtype=torch.float32)
        take = min(len(residual), target_tokens - tokens_seen)
        calibration_residual = residual[:take]
        if subtract_pre_bias and pre_bias is None:
            pre_bias = geometric_median(calibration_residual)
        squared_norm_sum += calibration_residual.square().sum().item()
        tokens_seen += take
        progress.update(take)
        if tokens_seen >= target_tokens:
            break
    progress.close()
    if tokens_seen == 0:
        raise ValueError("raw-L2 activation filter rejected every normalization token")

    mean_squared_norm = squared_norm_sum / tokens_seen
    scale = math.sqrt(d_model / mean_squared_norm)
    if pre_bias is not None:
        pre_bias.mul_(scale)
    return scale, pre_bias


def train_sae(
    sae: TopKSAE,
    train_batches: ResidualBatchFactory,
    validation_batches: ResidualBatchFactory,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: Path,
    resume: bool,
    log_every: int,
    checkpoint_every: int,
) -> dict:
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    checkpoint_metrics_path = output_dir / "checkpoint_metrics.jsonl"
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
        checkpoint_metrics_path.write_text("")
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

    def evaluate_checkpoint() -> dict:
        nonlocal evaluation_seconds
        evaluation_start = time.monotonic()
        evaluation = evaluate_sae(
            sae,
            validation_batches,
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

    if resume:
        latest_evaluation = load_checkpoint_evaluation(
            checkpoint_metrics_path, state.processed_tokens
        )
        if latest_evaluation is None:
            latest_evaluation = evaluate_checkpoint()

    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    next_log = ((state.processed_tokens // log_every) + 1) * log_every
    next_checkpoint = (
        (state.processed_tokens // checkpoint_every) + 1
    ) * checkpoint_every
    progress = tqdm(
        total=config.train_tokens,
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
    last_evaluated_training_tokens = (
        state.processed_tokens if latest_evaluation is not None else None
    )
    for residual in train_batches(skip_batches=state.processed_batches):
        batch_index = state.processed_batches
        state.processed_batches += 1
        residual = filter_raw_l2_activations(residual, config.max_activation_l2)
        if len(residual) == 0:
            continue
        residual = residual.to(device=device, dtype=torch.float32)
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
            or state.processed_tokens == config.train_tokens
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

        if (
            state.processed_tokens >= next_checkpoint
            or state.processed_tokens == config.train_tokens
        ):
            save_checkpoint(checkpoint_path, sae, optimizer, state, config)
            latest_evaluation = evaluate_checkpoint()
            last_evaluated_training_tokens = state.processed_tokens
            while next_checkpoint <= state.processed_tokens:
                next_checkpoint += checkpoint_every

    metric_status.close()
    progress.close()
    if last_evaluated_training_tokens != state.processed_tokens:
        latest_evaluation = evaluate_checkpoint()
    torch.save(
        {"sae": sae.state_dict()},
        output_dir / "sae_final.pt",
    )
    return latest_evaluation


@torch.inference_mode()
def evaluate_sae(
    sae: TopKSAE,
    validation_batches: ResidualBatchFactory,
    device: torch.device,
    config: ExperimentConfig,
) -> dict:
    metrics = RunningMetrics(sae.d_model, sae.d_sae, device)
    progress = tqdm(
        total=config.validation_tokens,
        unit="tok",
        desc="Validate",
        leave=False,
        disable=None,
    )
    evaluated_tokens = 0
    for residual in validation_batches(skip_batches=0):
        residual = filter_raw_l2_activations(residual, config.max_activation_l2)
        if len(residual) == 0:
            continue
        residual = residual.to(device=device, dtype=torch.float32)
        residual.mul_(config.activation_scale)
        batch_tokens = len(residual)
        evaluated_tokens += batch_tokens
        for start in range(0, len(residual), config.sae_batch_size):
            x = residual[start : start + config.sae_batch_size]
            reconstruction, indices, values = sae(x)
            metrics.update(x, reconstruction, indices, values)
        progress.update(batch_tokens)
    progress.close()
    if evaluated_tokens == 0:
        raise ValueError("raw-L2 activation filter rejected every validation token")

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

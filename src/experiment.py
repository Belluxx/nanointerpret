import json
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from .runtime import patch_layer_input
from .sae import FIRING_THRESHOLD, RunningMetrics, TopKSAE, auxk_loss

DOWNSTREAM_KL_TOKENS = 1_000_000
# Dimensions whose std exceeds this multiple of the median dimension's std are
# "massive activations" (attention-sink channels) and bypass the SAE unchanged.
PASSTHROUGH_STD_RATIO = 15.0


@dataclass(frozen=True)
class Config:
    model_id: str
    model_dtype: str
    layer_index: int
    d_model: int
    dataset_id: str
    dataset_config: str
    context_size: int
    train_tokens: int
    validation_tokens: int
    recording_tokens: int
    width_multiplier: int
    k: int
    aux_k: int
    aux_k_coef: float
    subtract_pre_bias: bool
    learning_rate: float
    gradient_clip: float | None
    dead_window: int
    model_batch_size: int
    sae_batch_size: int
    normalization_tokens: int
    passthrough_dims: list[int]
    activation_scale: float
    seed: int


def build_sae(config: Config) -> TopKSAE:
    return TopKSAE(
        config.d_model,
        config.width_multiplier * config.d_model,
        config.k,
        passthrough_dims=config.passthrough_dims,
        activation_scale=config.activation_scale,
        subtract_pre_bias=config.subtract_pre_bias,
    )


def load_sae(sae_dir: Path, device: torch.device) -> tuple[TopKSAE, Config]:
    config = Config(**json.loads((sae_dir / "config.json").read_text()))
    sae = build_sae(config)
    sae.load_state_dict(torch.load(sae_dir / "sae_final.pt", map_location="cpu"))
    return sae.to(device).eval().requires_grad_(False), config


@torch.inference_mode()
def calibrate(batches: Iterable[Tensor], token_count: int, detect_passthrough: bool) -> tuple[list[int], float]:
    # Pick the pass-through dims, then the scale giving the rest a mean squared norm equal to their width.
    seen = 0
    dim_sum = dim_sq_sum = 0
    with tqdm(total=token_count, unit="tok", desc="Calibrate", leave=False, dynamic_ncols=True) as progress:
        for residual in batches:
            # Float64 avoids cancellation on dims with large constant offsets. MPS silently
            # zeroes a direct float64 transfer, so move to the CPU first.
            residual = residual[: token_count - seen].cpu().double()
            dim_sum = dim_sum + residual.sum(dim=0)
            dim_sq_sum = dim_sq_sum + residual.square().sum(dim=0)
            seen += len(residual)
            progress.update(len(residual))
            if seen >= token_count:
                break

    passthrough_dims = []
    if detect_passthrough:
        dim_std = (dim_sq_sum / seen - (dim_sum / seen).square()).sqrt()
        std_ratio = dim_std / dim_std.median()
        passthrough_dims = torch.nonzero(std_ratio > PASSTHROUGH_STD_RATIO).flatten().tolist()
        print(
            f"Pass-through dims: {passthrough_dims or 'none'}"
            + "".join(f" | {dim}: {std_ratio[dim]:.0f}x median std" for dim in passthrough_dims)
        )
    keep = torch.ones(len(dim_sum), dtype=torch.bool)
    keep[passthrough_dims] = False
    return passthrough_dims, math.sqrt(keep.sum().item() * seen / dim_sq_sum[keep].sum().item())


@torch.inference_mode()
def evaluate_sae(sae: TopKSAE, batches: Iterable[Tensor], token_count: int, sae_batch_size: int) -> dict:
    metrics = RunningMetrics(sae)
    with tqdm(total=token_count, unit="tok", desc="Validate", leave=False, disable=None) as progress:
        for residual in batches:
            for x in sae.normalize(residual).split(sae_batch_size):
                reconstruction, indices, values, _ = sae(x)
                metrics.update(x, reconstruction, indices, values)
            progress.update(len(residual))

    fire_counts = metrics.fire_counts
    return {
        "tokens": metrics.count,
        **metrics.compute(),
        "dead_feature_pct": 100.0 * (fire_counts == 0).float().mean().item(),
        "active_features": int((fire_counts > 0).sum().item()),
        **feature_density_histogram(fire_counts, metrics.count),
    }


def feature_density_histogram(fire_counts: Tensor, token_count: int) -> dict:
    density = fire_counts[fire_counts > 0].cpu().double().numpy() / token_count
    min_exponent = -math.ceil(math.log10(token_count))
    bin_counts, bin_edges = np.histogram(np.log10(density), bins=np.linspace(min_exponent, 0.0, -min_exponent * 10 + 1))
    return {
        "total_features": fire_counts.numel(),
        "feature_density_log10_bin_edges": bin_edges.tolist(),
        "feature_density_bin_counts": bin_counts.tolist(),
    }


@torch.inference_mode()
def downstream_kl(sae: TopKSAE, model: nn.Module, layer: nn.Module, contexts: np.ndarray) -> dict:
    # Mean KL(base || SAE) over next-token predictions, with a 95% CI over contexts.
    contexts = contexts[: DOWNSTREAM_KL_TOKENS // contexts.shape[1]]
    context_kls = []
    # One context at a time keeps the full-vocabulary logits memory-bounded.
    for context in tqdm(contexts, unit="ctx", desc="Downstream KL", leave=False, disable=None):
        input_ids = torch.from_numpy(context[None].astype(np.int64)).to(model.device)
        base_logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        with patch_layer_input(layer, sae.reconstruct):
            sae_logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        base_log_z = torch.logsumexp(base_logits, dim=-1)
        sae_log_z = torch.logsumexp(sae_logits, dim=-1)

        # Computed in place over the logits to avoid another vocab-sized buffer.
        sae_logits.neg_().add_(base_logits)
        base_logits.sub_(base_log_z.unsqueeze(-1)).exp_()
        token_kl = sae_logits.mul_(base_logits).sum(dim=-1).add_(sae_log_z).sub_(base_log_z)
        context_kls.append(token_kl.mean().item())

    # Every context has the same number of predictions, so this is the token mean.
    mean_kl = float(np.mean(context_kls))
    margin = 1.96 * float(np.std(context_kls, ddof=1)) / math.sqrt(len(context_kls))
    return {"downstream_kl": mean_kl, "downstream_kl_ci": [max(0.0, mean_kl - margin), mean_kl + margin]}


def train_step(
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    last_fired: Tensor,
    position: int,
    config: Config,
    metrics: RunningMetrics,
) -> None:
    reconstruction, indices, values, pre_activations = sae(x)
    last_fired[indices[values > FIRING_THRESHOLD]] = position
    loss = F.mse_loss(reconstruction, x)
    aux = None
    if config.aux_k_coef > 0 and position >= config.dead_window:
        dead = torch.nonzero(last_fired < position - config.dead_window).flatten()
        if len(dead):
            aux = auxk_loss(sae, pre_activations, x - reconstruction, dead, config.aux_k)
            loss = loss + config.aux_k_coef * aux
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    sae.constrain_decoder_gradient()
    if config.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(sae.parameters(), config.gradient_clip)
    optimizer.step()
    sae.normalize_decoder()
    metrics.update(x, reconstruction.detach(), indices, values, aux)


def train_sae(
    sae: TopKSAE,
    config: Config,
    train_batches: Callable[[int], Iterator[Tensor]],
    evaluate: Callable[[], dict],
    output_dir: Path,
    resume: bool,
    log_every: int,
    checkpoint_every: int,
    validate_every: int,
) -> None:
    # `train_batches(skip)` yields the shuffled training residuals, skipping `skip` contexts.
    device = sae.decoder_bias.device
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    train_log = output_dir / "train_metrics.jsonl"
    evaluation_log = output_dir / "evaluation_metrics.jsonl"
    tokens = 0
    last_fired = torch.full((sae.d_sae,), -1, device=device)
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        sae.load_state_dict(checkpoint["sae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        tokens, last_fired = checkpoint["tokens"], checkpoint["last_fired"]
        truncate_jsonl(train_log, "tokens", tokens)
        truncate_jsonl(evaluation_log, "training_tokens", tokens)
        print(f"Resumed at {tokens:,} training tokens")
    else:
        train_log.write_text("")
        evaluation_log.write_text("")

    total_tokens = config.train_tokens // config.context_size * config.context_size
    metrics = RunningMetrics(sae)
    progress = tqdm(total=total_tokens, initial=tokens, unit="tok", desc="Train", dynamic_ncols=True, disable=None)
    status = tqdm(desc="Metrics", bar_format="{desc}", dynamic_ncols=True, disable=progress.disable)
    start_time, start_tokens, evaluation_seconds = time.monotonic(), tokens, 0.0
    for residual in train_batches(tokens // config.context_size):
        x = sae.normalize(residual)
        # Mix tokens across contexts before splitting into SAE batches, reproducibly across resumes.
        x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(config.seed + tokens)).to(device)]
        position = tokens
        for x_batch in x.split(config.sae_batch_size):
            position += len(x_batch)
            train_step(sae, optimizer, x_batch, last_fired, position, config, metrics)
        previous, tokens = tokens, position
        progress.update(tokens - previous)
        done = tokens == total_tokens

        if done or crossed(previous, tokens, log_every):
            dead = last_fired < tokens - config.dead_window
            record = {
                "tokens": tokens,
                **metrics.compute(),
                "dead_feature_pct": 100.0 * dead.float().mean().item() if tokens >= config.dead_window else None,
                "tokens_per_second": (tokens - start_tokens) / (time.monotonic() - start_time - evaluation_seconds),
            }
            append_jsonl(train_log, record)
            if progress.disable:
                print(f"Train: {tokens:,} tok | {format_metrics(record)}")
            else:
                status.set_description_str(f"Metrics\t{format_metrics(record)}")
            metrics = RunningMetrics(sae)

        if done or crossed(previous, tokens, validate_every):
            evaluation_start = time.monotonic()
            record = {"training_tokens": tokens, **evaluate()}
            evaluation_seconds += time.monotonic() - evaluation_start
            append_jsonl(evaluation_log, record)
            tqdm.write(f"Validation: {tokens:,} tok | {format_metrics(record)}")

        if crossed(previous, tokens, checkpoint_every):
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {"sae": sae.state_dict(), "optimizer": optimizer.state_dict(), "tokens": tokens, "last_fired": last_fired},
                temporary,
            )
            os.replace(temporary, checkpoint_path)

    status.close()
    progress.close()
    torch.save(sae.state_dict(), output_dir / "sae_final.pt")


def crossed(previous: int, current: int, every: int) -> bool:
    return current // every > previous // every


def format_metrics(record: dict) -> str:
    parts = []
    if "downstream_kl" in record:
        parts.append(f"downstream KL {record['downstream_kl']:.6g}")
    parts += [f"EV {record['explained_variance']:.2%}", f"MSE {record['mse']:,.4f}"]
    if record["auxk_loss"] is not None:
        parts.append(f"AuxK NMSE {record['auxk_loss']:,.4f}")
    if record["dead_feature_pct"] is not None:
        parts.append(f"dead {record['dead_feature_pct']:.2f}%")
    return " | ".join(parts)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def truncate_jsonl(path: Path, key: str, limit: int) -> None:
    # Drop records logged after the checkpoint being resumed.
    lines = [line for line in path.read_text().splitlines() if json.loads(line)[key] <= limit]
    path.write_text("".join(line + "\n" for line in lines))

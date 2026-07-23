"""Find SAE features that activate strongly across diverse target tokens."""

from __future__ import annotations

import argparse
import heapq
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, Iterator

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import iter_context_batches, token_cache_paths
from src.experiment import (
    ResidualStreamCapture,
    find_transformer_layers,
    move_and_capture_residual,
)
from src.sae import TopKSAE


@dataclass(frozen=True)
class ActivationBatch:
    indices: Tensor
    values: Tensor


@dataclass(frozen=True)
class TokenActivationBatch(ActivationBatch):
    positions: Tensor
    token_ids: Tensor


@dataclass(frozen=True)
class Sample:
    activation: float
    position: int
    token_id: int
    token: str


@dataclass(frozen=True)
class FeatureReport:
    index: int
    fire_count: int
    frequency: float
    score: float
    diversity_score: float
    activation_score: float
    unique_strong_tokens: int
    mean_sample_activation: float
    max_activation: float
    samples: tuple[Sample, ...]


class DiverseSamplePool:
    """Keep high-activation examples without letting one token dominate."""

    def __init__(self, token_limit: int, samples_per_token: int, context_size: int):
        self.token_limit = token_limit
        self.samples_per_token = samples_per_token
        self.context_size = context_size
        self.by_token: dict[str, dict[int, Sample]] = {}
        self.best_by_token: dict[str, float] = {}
        self.weakest_tokens: list[tuple[float, str]] = []

    def _context_id(self, sample: Sample) -> int:
        return sample.position // self.context_size

    def _discard_stale_tokens(self) -> None:
        while self.weakest_tokens:
            activation, token = self.weakest_tokens[0]
            if self.best_by_token.get(token) == activation:
                break
            heapq.heappop(self.weakest_tokens)

    def _update_best(self, token: str, contexts: dict[int, Sample]) -> None:
        best = max(sample.activation for sample in contexts.values())
        if self.best_by_token.get(token) != best:
            self.best_by_token[token] = best
            heapq.heappush(self.weakest_tokens, (best, token))

    def add(self, sample: Sample) -> None:
        contexts = self.by_token.get(sample.token)
        if contexts is None:
            if len(self.by_token) >= self.token_limit:
                self._discard_stale_tokens()
                if self.weakest_tokens and sample.activation <= self.weakest_tokens[0][0]:
                    return
                _activation, weakest = heapq.heappop(self.weakest_tokens)
                del self.by_token[weakest]
                del self.best_by_token[weakest]
            context_id = self._context_id(sample)
            contexts = {context_id: sample}
            self.by_token[sample.token] = contexts
            self._update_best(sample.token, contexts)
            return

        context_id = self._context_id(sample)
        previous = contexts.get(context_id)
        if previous is not None:
            if sample.activation > previous.activation:
                contexts[context_id] = sample
        elif len(contexts) < self.samples_per_token:
            contexts[context_id] = sample
        else:
            weakest_context, weakest = min(
                contexts.items(), key=lambda item: item[1].activation
            )
            if sample.activation > weakest.activation:
                del contexts[weakest_context]
                contexts[context_id] = sample

        self._update_best(sample.token, contexts)

    def strong_tokens(self, relative_threshold: float) -> list[tuple[str, list[Sample]]]:
        if not self.best_by_token:
            return []
        maximum = max(self.best_by_token.values())
        threshold = maximum * relative_threshold
        result = []
        for token, contexts in self.by_token.items():
            samples = sorted(
                contexts.values(), key=lambda sample: sample.activation, reverse=True
            )
            if samples[0].activation >= threshold:
                result.append((token, samples))
        return sorted(result, key=lambda item: item[1][0].activation, reverse=True)

    def diversified_samples(
        self, limit: int, relative_threshold: float
    ) -> tuple[list[Sample], int]:
        strong = self.strong_tokens(relative_threshold)
        selected: list[Sample] = []
        used_contexts: set[int] = set()

        # First show one example per distinct token. Prefer a fresh context, but never
        # sacrifice token diversity merely because two tokens occur in the same context.
        for _token, samples in strong:
            sample = next(
                (
                    candidate
                    for candidate in samples
                    if self._context_id(candidate) not in used_contexts
                ),
                samples[0],
            )
            selected.append(sample)
            used_contexts.add(self._context_id(sample))
            if len(selected) == limit:
                return selected, len(strong)

        # Then fill remaining space with different contexts, still limiting each token.
        used_positions = {sample.position for sample in selected}
        remaining = sorted(
            (
                sample
                for _token, samples in strong
                for sample in samples
                if sample.position not in used_positions
            ),
            key=lambda sample: sample.activation,
            reverse=True,
        )
        for sample in remaining:
            context_id = self._context_id(sample)
            if context_id in used_contexts:
                continue
            selected.append(sample)
            used_contexts.add(context_id)
            if len(selected) == limit:
                break
        return selected, len(strong)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"))
    parser.add_argument(
        "--tokens-path",
        type=Path,
        help="Explicit uint32 token file to scan instead of the checkpoint's validation cache.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Defaults to feature_report.md beside the checkpoint.",
    )
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--samples-per-feature", type=int, default=10)
    parser.add_argument("--minimum-fires", type=int, default=100)
    parser.add_argument("--minimum-frequency", type=float, default=1e-4)
    parser.add_argument("--maximum-frequency", type=float, default=1e-2)
    parser.add_argument("--minimum-unique-tokens", type=int, default=5)
    parser.add_argument("--candidate-features", type=int, default=4096)
    parser.add_argument("--candidate-tokens-per-feature", type=int, default=64)
    parser.add_argument("--samples-per-token", type=int, default=2)
    parser.add_argument(
        "--minimum-relative-activation",
        type=float,
        default=0.5,
        help="A token counts as diverse only if its best activation reaches this fraction of the feature maximum.",
    )
    parser.add_argument("--context-tokens", type=int, default=20)
    parser.add_argument(
        "--scan-tokens",
        type=int,
        help="Number of validation tokens to scan. Defaults to the complete cache.",
    )
    parser.add_argument("--model-batch-size", type=int)
    parser.add_argument("--sae-batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        help="Defaults to the dtype stored in the checkpoint.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.features,
        args.samples_per_feature,
        args.minimum_fires,
        args.minimum_unique_tokens,
        args.candidate_features,
        args.candidate_tokens_per_feature,
        args.samples_per_token,
        args.context_tokens,
    )
    optional = (args.scan_tokens, args.model_batch_size, args.sae_batch_size)
    if min(positive) <= 0 or any(value is not None and value <= 0 for value in optional):
        raise ValueError("counts and batch sizes must be positive")
    if not 0 < args.minimum_frequency < args.maximum_frequency <= 1:
        raise ValueError("frequencies must satisfy 0 < minimum < maximum <= 1")
    if not 0 < args.minimum_relative_activation <= 1:
        raise ValueError("minimum relative activation must be in (0, 1]")
    if args.minimum_unique_tokens > args.candidate_tokens_per_feature:
        raise ValueError("minimum unique tokens cannot exceed candidate tokens per feature")


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


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("sae") if isinstance(checkpoint, dict) else None
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError(f"{path} is not a valid SAE checkpoint")

    required = {
        "model_id",
        "dataset_config",
        "train_tokens",
        "validation_tokens",
        "context_size",
        "layer_index",
        "d_model",
        "d_sae",
        "k",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"checkpoint configuration is missing: {', '.join(missing)}")
    return state, config


def find_validation_cache(config: dict, cache_dir: Path) -> Path:
    cache_args = SimpleNamespace(cache_dir=cache_dir, **config)
    _train, validation, _metadata = token_cache_paths(cache_args)
    if not validation.exists():
        raise FileNotFoundError(
            f"validation cache not found at {validation}; run train.py --cache-only first"
        )
    return validation


@torch.inference_mode()
def iter_feature_activations(
    model,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    tokens: np.ndarray,
    pad_token_id: int,
    context_size: int,
    model_batch_size: int,
    sae_batch_size: int,
    device: torch.device,
    description: str,
) -> Iterator[ActivationBatch]:
    batches = iter_context_batches(
        tokens,
        context_size,
        model_batch_size,
        pad_token_id,
        shuffle=False,
        seed=0,
    )
    progress = tqdm(total=len(tokens), unit="tok", desc=description, dynamic_ncols=True)
    try:
        for input_ids, attention_mask, _ in batches:
            token_count = int(attention_mask.sum())
            residual, _, _ = move_and_capture_residual(
                model, capture, input_ids, attention_mask, token_count, device
            )

            for start in range(0, len(residual), sae_batch_size):
                indices, values = sae.encode(residual[start : start + sae_batch_size])
                yield ActivationBatch(indices, values)
            progress.update(token_count)
    finally:
        progress.close()


def collect_feature_stats(
    activation_batches: Iterator[ActivationBatch],
    feature_count: int,
    index_cache: BinaryIO,
    value_cache: BinaryIO,
    index_dtype: np.dtype | type[np.unsignedinteger],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(feature_count, dtype=np.int64)
    activation_sums = np.zeros(feature_count, dtype=np.float64)
    activation_square_sums = np.zeros(feature_count, dtype=np.float64)
    for batch in activation_batches:
        indices = batch.indices.cpu().numpy()
        values = batch.values.float().cpu().numpy()
        indices.astype(index_dtype, copy=False).tofile(index_cache)
        values.tofile(value_cache)

        positive = values > 0
        indices = indices[positive]
        values = values[positive]
        counts += np.bincount(indices, minlength=feature_count)
        activation_sums += np.bincount(indices, weights=values, minlength=feature_count)
        activation_square_sums += np.bincount(
            indices, weights=values * values, minlength=feature_count
        )
    return counts, activation_sums, activation_square_sums


def iter_cached_feature_activations(
    index_cache: BinaryIO,
    value_cache: BinaryIO,
    tokens: np.ndarray,
    k: int,
    batch_size: int,
    index_dtype: np.dtype | type[np.unsignedinteger],
) -> Iterator[TokenActivationBatch]:
    """Replay a compact activation stream without rerunning the model and SAE."""
    index_cache.seek(0)
    value_cache.seek(0)
    progress = tqdm(
        total=len(tokens), unit="tok", desc="Rank diverse tokens", dynamic_ncols=True
    )
    try:
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            shape = (end - start, k)
            count = shape[0] * shape[1]
            indices = np.fromfile(index_cache, dtype=index_dtype, count=count)
            values = np.fromfile(value_cache, dtype=np.float32, count=count)
            if len(indices) != count or len(values) != count:
                raise RuntimeError("temporary activation cache is incomplete")
            yield TokenActivationBatch(
                indices=torch.from_numpy(indices.astype(np.int64).reshape(shape)),
                values=torch.from_numpy(values.reshape(shape)),
                positions=torch.arange(start, end),
                token_ids=torch.from_numpy(np.asarray(tokens[start:end], dtype=np.int64)),
            )
            progress.update(end - start)
    finally:
        progress.close()


def select_candidate_features(
    counts: np.ndarray,
    activation_sums: np.ndarray,
    activation_square_sums: np.ndarray,
    token_count: int,
    args: argparse.Namespace,
) -> np.ndarray:
    frequencies = counts / token_count
    eligible = (
        (counts >= args.minimum_fires)
        & (frequencies >= args.minimum_frequency)
        & (frequencies <= args.maximum_frequency)
    )
    candidates = np.flatnonzero(eligible)
    means = activation_sums[candidates] / counts[candidates]
    rms = np.sqrt(activation_square_sums[candidates] / counts[candidates])
    strength = np.sqrt(means * rms)
    order = np.argsort(-strength, kind="stable")
    return candidates[order[: args.candidate_features]]


def normalized_token(tokenizer, token_id: int, cache: dict[int, str | None]) -> str | None:
    if token_id in cache:
        return cache[token_id]
    if token_id in tokenizer.all_special_ids:
        normalized = None
    else:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        normalized = decoded.casefold()
        # Pure digits trivially look diverse by covering 0-9, but do not make a
        # feature linguistically interesting. Alphanumeric technical tokens remain.
        if not normalized or not any(character.isalpha() for character in normalized):
            normalized = None
    cache[token_id] = normalized
    return normalized


def collect_diverse_pools(
    activation_batches: Iterator[TokenActivationBatch],
    candidate_features: np.ndarray,
    feature_count: int,
    tokenizer,
    args: argparse.Namespace,
    context_size: int,
) -> dict[int, DiverseSamplePool]:
    pools = {
        int(index): DiverseSamplePool(
            args.candidate_tokens_per_feature, args.samples_per_token, context_size
        )
        for index in candidate_features
    }
    if not pools:
        return pools

    lookup = torch.full((feature_count,), -1, dtype=torch.int64)
    candidate_ids = torch.tensor(list(pools), dtype=torch.int64)
    lookup[candidate_ids] = candidate_ids
    token_cache: dict[int, str | None] = {}

    for batch in activation_batches:
        feature_indices = lookup[batch.indices]
        rows, columns = torch.where((feature_indices >= 0) & (batch.values > 0))
        if rows.numel() == 0:
            continue

        event_features = feature_indices[rows, columns].cpu().numpy()
        event_values = batch.values[rows, columns].float().cpu().numpy()
        event_token_ids = batch.token_ids[rows].cpu().numpy()
        event_positions = batch.positions[rows.cpu()].numpy()

        # Retain the strongest occurrence of each feature/token pair per batch.
        keys = event_features * len(tokenizer) + event_token_ids
        order = np.lexsort((-event_values, keys))
        sorted_keys = keys[order]
        first = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))

        for event in order[first]:
            token_id = int(event_token_ids[event])
            token = normalized_token(tokenizer, token_id, token_cache)
            if token is None:
                continue
            feature_id = int(event_features[event])
            pools[feature_id].add(
                Sample(
                    activation=float(event_values[event]),
                    position=int(event_positions[event]),
                    token_id=token_id,
                    token=token,
                )
            )
    return pools


def build_feature_reports(
    pools: dict[int, DiverseSamplePool],
    counts: np.ndarray,
    token_count: int,
    args: argparse.Namespace,
) -> list[FeatureReport]:
    candidates: list[FeatureReport] = []
    for feature_id, pool in pools.items():
        samples, unique_tokens = pool.diversified_samples(
            args.samples_per_feature, args.minimum_relative_activation
        )
        if unique_tokens < args.minimum_unique_tokens or not samples:
            continue
        first_per_token: dict[str, Sample] = {}
        for sample in samples:
            first_per_token.setdefault(sample.token, sample)
        distinct_samples = list(first_per_token.values())
        activations = np.asarray(
            [sample.activation for sample in distinct_samples], dtype=np.float64
        )
        mean_activation = float(activations.mean())
        maximum = max(pool.best_by_token.values())
        coverage = min(unique_tokens / args.samples_per_feature, 1.0)
        balance = mean_activation / maximum
        candidates.append(
            FeatureReport(
                index=feature_id,
                fire_count=int(counts[feature_id]),
                frequency=float(counts[feature_id] / token_count),
                score=0.0,
                diversity_score=coverage * balance,
                activation_score=0.0,
                unique_strong_tokens=unique_tokens,
                mean_sample_activation=mean_activation,
                max_activation=float(maximum),
                samples=tuple(samples),
            )
        )

    if not candidates:
        return []

    log_strengths = np.log1p(
        [candidate.mean_sample_activation for candidate in candidates]
    )
    low, high = np.percentile(log_strengths, (10, 90))
    scale = max(float(high - low), 1e-12)
    reports = []
    for candidate, log_strength in zip(candidates, log_strengths):
        activation_score = float(np.clip((log_strength - low) / scale, 0.0, 1.0))
        score = 0.7 * candidate.diversity_score + 0.3 * activation_score
        reports.append(
            replace(
                candidate,
                score=score,
                activation_score=activation_score,
            )
        )
    reports.sort(key=lambda feature: (-feature.score, feature.index))
    return reports[: args.features]


def highlighted_context(
    tokenizer,
    tokens: np.ndarray,
    position: int,
    radius: int,
    context_size: int,
) -> str:
    context_start = position // context_size * context_size
    start = max(context_start, position - radius)
    end = min(context_start + context_size, position + radius + 1, len(tokens))
    ids = [int(token) for token in tokens[start:end]]
    target = position - start
    options = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    prefix = tokenizer.decode(ids[:target], **options)
    token = tokenizer.decode([ids[target]], **options)
    suffix = tokenizer.decode(ids[target + 1 :], **options)
    if not token:
        token = str(tokenizer.convert_ids_to_tokens(ids[target]))
    context = f"{prefix}<<{token}>>{suffix}"
    return context.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def inline_code(value: str) -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{fence}{padding}{value}{padding}{fence}"


def render_report(
    checkpoint: Path,
    config: dict,
    tokenizer,
    tokens: np.ndarray,
    features: list[FeatureReport],
    candidate_count: int,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Interesting SAE features",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Model: `{config['model_id']}`",
        f"- Validation tokens scanned: {len(tokens):,}",
        f"- Strong candidates inspected: {candidate_count:,}",
        f"- Frequency range: `{args.minimum_frequency:g}` to `{args.maximum_frequency:g}`",
        f"- Strong-token cutoff: `{args.minimum_relative_activation:.0%}` of feature maximum",
        "",
        "Features are ranked primarily by strong normalized-token diversity, then by activation strength. "
        "Repeated case variants, numeric-only tokens, punctuation, special tokens, and duplicate "
        "contexts are suppressed.",
        "",
    ]
    if not features:
        lines.append("No features satisfied the interestingness thresholds.")
        return "\n".join(lines) + "\n"

    context_size = int(config["context_size"])
    for rank, feature in enumerate(features, 1):
        token_preview = ", ".join(
            dict.fromkeys(sample.token for sample in feature.samples)
        )
        lines += [
            f"## {rank}. Feature {feature.index}",
            "",
            f"- Interestingness: `{feature.score:.3f}` "
            f"(diversity `{feature.diversity_score:.3f}`, activation `{feature.activation_score:.3f}`)",
            f"- Fires: {feature.fire_count:,} / {len(tokens):,} tokens "
            f"(`{feature.frequency:.6%}`)",
            f"- Strong distinct tokens: {feature.unique_strong_tokens}",
            f"- Sample activation: mean `{feature.mean_sample_activation:.5g}`, "
            f"maximum `{feature.max_activation:.5g}`",
            f"- Sampled tokens: {inline_code(token_preview)}",
            "",
        ]
        for number, sample in enumerate(feature.samples, 1):
            context = highlighted_context(
                tokenizer,
                tokens,
                sample.position,
                args.context_tokens,
                context_size,
            )
            lines.append(
                f"{number}. activation `{sample.activation:.5g}`, token `{sample.token}`, "
                f"token id `{sample.token_id}`, position `{sample.position}` — "
                f"{inline_code(context)}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    checkpoint = args.checkpoint
    state, config = load_checkpoint(checkpoint)

    cache_path = (
        args.tokens_path
        if args.tokens_path is not None
        else find_validation_cache(config, args.cache_dir)
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"token file not found at {cache_path}")
    cached_tokens = np.memmap(cache_path, mode="r", dtype=np.uint32)
    token_count = min(args.scan_tokens or len(cached_tokens), len(cached_tokens))
    tokens = cached_tokens[:token_count]
    context_size = int(config["context_size"])
    model_batch_size = args.model_batch_size or int(config.get("model_batch_size", 32))
    sae_batch_size = args.sae_batch_size or int(config.get("sae_batch_size", 8192))
    dtype = getattr(torch, args.model_dtype or config.get("model_dtype", "float32"))

    print(f"device={device}, tokens={len(tokens):,}, checkpoint={checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval().requires_grad_(False)

    layer_path, layers = find_transformer_layers(model)
    layer_index = int(config["layer_index"])
    if config.get("layer_path", layer_path) != layer_path:
        raise ValueError(f"model layer path changed from {config['layer_path']} to {layer_path}")
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer index must be in [0, {len(layers) - 1}]")

    sae = TopKSAE(
        int(config["d_model"]), int(config["d_sae"]), int(config["k"]), device
    )
    sae.load_state_dict(state)
    sae.eval().requires_grad_(False)

    index_dtype = (
        np.uint16 if sae.d_sae <= np.iinfo(np.uint16).max + 1 else np.uint32
    )
    with (
        tempfile.TemporaryFile() as index_cache,
        tempfile.TemporaryFile() as value_cache,
        ResidualStreamCapture(layers[layer_index]) as capture,
    ):
        activation_batches = iter_feature_activations(
            model,
            capture,
            sae,
            tokens,
            tokenizer.pad_token_id,
            context_size,
            model_batch_size,
            sae_batch_size,
            device,
            "Scan features",
        )
        counts, activation_sums, activation_square_sums = collect_feature_stats(
            activation_batches,
            sae.d_sae,
            index_cache,
            value_cache,
            index_dtype,
        )
        candidates = select_candidate_features(
            counts,
            activation_sums,
            activation_square_sums,
            len(tokens),
            args,
        )
        pools = collect_diverse_pools(
            iter_cached_feature_activations(
                index_cache,
                value_cache,
                tokens,
                sae.k,
                sae_batch_size,
                index_dtype,
            ),
            candidates,
            sae.d_sae,
            tokenizer,
            args,
            context_size,
        )

    features = build_feature_reports(pools, counts, len(tokens), args)
    report = render_report(
        checkpoint, config, tokenizer, tokens, features, len(candidates), args
    )
    report_path = args.report_path or checkpoint.parent / "feature_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {report_path} ({len(features)} interesting features)")


if __name__ == "__main__":
    main()

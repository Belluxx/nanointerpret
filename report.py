"""Report rare, overactive, and typical SAE features with token examples."""

from __future__ import annotations

import argparse
import heapq
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

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
from src.sae import RunningMetrics, TopKSAE


CATEGORIES = ("rare", "overactive", "other")
CATEGORY_TITLES = {
    "rare": "Rare features",
    "overactive": "Overactive features",
    "other": "Other features",
}


@dataclass(frozen=True)
class Feature:
    index: int
    category: str
    fire_count: int
    frequency: float


@dataclass(frozen=True)
class Sample:
    activation: float
    position: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/sae_gemma_3_270m")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to OUTPUT_DIR/sae_final.pt, then checkpoint.pt.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"))
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Defaults to OUTPUT_DIR/feature_report.md.",
    )
    parser.add_argument("--features-per-category", type=int, default=5)
    parser.add_argument("--samples-per-feature", type=int, default=10)
    parser.add_argument("--context-tokens", type=int, default=20)
    parser.add_argument(
        "--scan-tokens",
        type=int,
        help="Number of validation tokens to scan. Defaults to the complete cache.",
    )
    parser.add_argument(
        "--rare-frequency", type=float, default=RunningMetrics.RARE_FREQUENCY
    )
    parser.add_argument(
        "--overactive-frequency", type=float, default=RunningMetrics.OVERACTIVE_FREQUENCY
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
    values = [args.features_per_category, args.samples_per_feature, args.context_tokens]
    optional = (args.scan_tokens, args.model_batch_size, args.sae_batch_size)
    values += [value for value in optional if value is not None]
    if min(values) <= 0:
        raise ValueError("counts and batch sizes must be positive")
    if not 0 < args.rare_frequency < args.overactive_frequency <= 1:
        raise ValueError("frequencies must satisfy 0 < rare < overactive <= 1")


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


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    candidates = (
        [args.checkpoint]
        if args.checkpoint
        else [args.output_dir / "sae_final.pt", args.output_dir / "checkpoint.pt"]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("no SAE checkpoint found: " + ", ".join(map(str, candidates)))


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
            f"validation cache not found at {validation}; run main.py --cache-only first"
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
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
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
        for input_ids, attention_mask, context_ids in batches:
            token_count = int(attention_mask.sum())
            residual, _input_ids, _valid_mask = move_and_capture_residual(
                model, capture, input_ids, attention_mask, token_count, device
            )
            positions = (
                torch.as_tensor(context_ids)[:, None] * context_size
                + torch.arange(context_size)[None, :]
            )[attention_mask.bool()]

            for start in range(0, len(residual), sae_batch_size):
                batch = residual[start : start + sae_batch_size]
                indices, values = sae.encode(batch)
                yield indices, values, positions[start : start + len(batch)]
            progress.update(token_count)
    finally:
        progress.close()


def count_fires(activation_batches, feature_count: int) -> np.ndarray:
    counts = torch.zeros(feature_count, dtype=torch.int64)
    for indices, values, _positions in activation_batches:
        fired = indices[values > 0].cpu()
        counts += torch.bincount(fired, minlength=feature_count)
    return counts.numpy()


def select_features(
    counts: np.ndarray,
    token_count: int,
    per_category: int,
    minimum_fires: int,
    rare_threshold: float,
    overactive_threshold: float,
) -> tuple[list[Feature], dict[str, int]]:
    frequencies = counts / token_count
    masks = {
        "rare": (counts > 0) & (frequencies < rare_threshold),
        "overactive": frequencies > overactive_threshold,
        "other": (frequencies >= rare_threshold)
        & (frequencies <= overactive_threshold),
    }
    totals = {category: int(mask.sum()) for category, mask in masks.items()}
    selected: list[Feature] = []

    for category in CATEGORIES:
        candidates = np.flatnonzero(masks[category] & (counts >= minimum_fires))
        if category == "other" and len(candidates):
            median = np.median(np.log(frequencies[candidates]))
            order = np.argsort(np.abs(np.log(frequencies[candidates]) - median), kind="stable")
        else:
            order = np.argsort(-counts[candidates], kind="stable")

        for index in candidates[order[:per_category]]:
            selected.append(
                Feature(
                    index=int(index),
                    category=category,
                    fire_count=int(counts[index]),
                    frequency=float(frequencies[index]),
                )
            )
    return selected, totals


def keep_top_sample(heap: list[tuple[float, int]], item: tuple[float, int], limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def collect_samples(
    activation_batches,
    features: list[Feature],
    feature_count: int,
    sample_count: int,
    device: torch.device,
) -> dict[int, list[Sample]]:
    heaps = {feature.index: [] for feature in features}
    if not features:
        return {}

    feature_ids = [feature.index for feature in features]
    lookup = torch.full((feature_count,), -1, dtype=torch.int64, device=device)
    lookup[torch.tensor(feature_ids, device=device)] = torch.arange(
        len(feature_ids), device=device
    )

    for indices, values, positions in activation_batches:
        selected = lookup[indices]
        for slot, feature_id in enumerate(feature_ids):
            rows, columns = torch.where((selected == slot) & (values > 0))
            if rows.numel() == 0:
                continue
            activations = values[rows, columns]
            top_values, top_indices = torch.topk(
                activations, min(sample_count, len(activations))
            )
            top_positions = positions[rows[top_indices].cpu()]
            for activation, position in zip(top_values.cpu(), top_positions):
                keep_top_sample(
                    heaps[feature_id],
                    (float(activation), int(position)),
                    sample_count,
                )

    return {
        feature_id: [Sample(*item) for item in sorted(heap, reverse=True)]
        for feature_id, heap in heaps.items()
    }


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


def inline_code(text: str) -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def render_report(
    checkpoint: Path,
    config: dict,
    tokenizer,
    tokens: np.ndarray,
    features: list[Feature],
    samples: dict[int, list[Sample]],
    totals: dict[str, int],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# SAE feature report",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Model: `{config['model_id']}`",
        f"- Validation tokens scanned: {len(tokens):,}",
        f"- Rare: `0 < frequency < {args.rare_frequency:g}`",
        f"- Overactive: `frequency > {args.overactive_frequency:g}`",
        f"- Live features: rare {totals['rare']:,}, overactive "
        f"{totals['overactive']:,}, other {totals['other']:,}",
        "",
        "Samples are ordered by activation; the scored token is marked as `<<token>>`.",
        "",
    ]

    for category in CATEGORIES:
        lines += [f"## {CATEGORY_TITLES[category]}", ""]
        category_features = [feature for feature in features if feature.category == category]
        if not category_features:
            lines += [
                f"No features in this category fired at least "
                f"{args.samples_per_feature} times.",
                "",
            ]
            continue

        for feature in category_features:
            lines += [
                f"### Feature {feature.index}",
                "",
                f"Fires: {feature.fire_count:,} / {len(tokens):,} tokens "
                f"({feature.frequency:.6%}).",
                "",
            ]
            for number, sample in enumerate(samples[feature.index], 1):
                context = highlighted_context(
                    tokenizer,
                    tokens,
                    sample.position,
                    args.context_tokens,
                    int(config["context_size"]),
                )
                lines.append(
                    f"{number}. activation `{sample.activation:.5g}`, "
                    f"token id `{int(tokens[sample.position])}`, position `{sample.position}` "
                    f"— {inline_code(context)}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    checkpoint = resolve_checkpoint(args)
    state, config = load_checkpoint(checkpoint)

    cache_path = find_validation_cache(config, args.cache_dir)
    cached_tokens = np.memmap(cache_path, mode="r", dtype=np.uint32)
    token_count = min(args.scan_tokens or len(cached_tokens), len(cached_tokens))
    tokens = cached_tokens[:token_count]
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

    def scan(capture, description):
        return iter_feature_activations(
            model,
            capture,
            sae,
            tokens,
            tokenizer.pad_token_id,
            int(config["context_size"]),
            model_batch_size,
            sae_batch_size,
            device,
            description,
        )

    with ResidualStreamCapture(layers[layer_index]) as capture:
        counts = count_fires(scan(capture, "Classify features"), sae.d_sae)
        features, totals = select_features(
            counts,
            len(tokens),
            args.features_per_category,
            args.samples_per_feature,
            args.rare_frequency,
            args.overactive_frequency,
        )
        samples = collect_samples(
            scan(capture, "Collect samples"),
            features,
            sae.d_sae,
            args.samples_per_feature,
            device,
        )

    report = render_report(
        checkpoint, config, tokenizer, tokens, features, samples, totals, args
    )
    report_path = args.report_path or args.output_dir / "feature_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {report_path} ({len(features)} features)")


if __name__ == "__main__":
    main()

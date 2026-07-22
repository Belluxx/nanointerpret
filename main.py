"""
Train a Top-K sparse autoencoder on Gemma's midpoint residual stream.

Defaults: Gemma 3 270M, FineWeb-Edu sample-10BT, 10M train tokens,
1M validation tokens, context length 256, 16x expansion, and K=32.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_token_cache
from src.experiment import (
    ExperimentConfig,
    ResidualStreamCapture,
    collect_top_examples,
    evaluate_sae,
    find_transformer_layers,
    rank_reports_by_token_coherence,
    train_sae,
    write_example_markdown,
)
from src.sae import TopKSAE


MODEL_ID = "google/gemma-3-270m"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--validation-tokens", type=int, default=1_000_000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--width-multiplier", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=32,
        help="Contexts processed together; lower this if memory is limited.",
    )
    parser.add_argument(
        "--sae-batch-size",
        type=int,
        default=8192,
        help="SAE token microbatch; lower this if memory is limited.",
    )
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        help="Save and evaluate a checkpoint after this many training tokens.",
    )
    parser.add_argument("--dead-window", type=int, default=1_000_000)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-layer",
        type=int,
        default=None,
        help="Layer whose input is captured. Default: len(transformer.layers) // 2.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sae_gemma_3_270m"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument("--report-features", type=int, default=64)
    parser.add_argument("--examples-per-feature", type=int, default=5)
    parser.add_argument("--example-radius", type=int, default=24)
    return parser.parse_args()


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
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_tokens,
        args.validation_tokens,
        args.context_size,
        args.model_batch_size,
        args.sae_batch_size,
    )
    if min(positive) <= 0:
        raise ValueError("token counts, context size, and batch sizes must be positive")
    if args.log_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("log and checkpoint intervals must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)
    dtype = getattr(torch, args.model_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device}, model dtype={dtype}, model batch={args.model_batch_size}, "
        f"SAE batch={args.sae_batch_size}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path, validation_path = build_token_cache(tokenizer, args)
    if args.cache_only:
        print(f"cached train tokens: {train_path}")
        print(f"cached validation tokens: {validation_path}")
        return

    train_tokens = np.memmap(train_path, mode="r", dtype=np.uint32)
    validation_tokens = np.memmap(validation_path, mode="r", dtype=np.uint32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval().requires_grad_(False)

    layer_path, layers = find_transformer_layers(model)
    layer_index = len(layers) // 2 if args.activation_layer is None else args.activation_layer
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"activation layer must be in [0, {len(layers) - 1}], got {layer_index}")

    d_model = int(model.config.hidden_size)
    d_sae = args.width_multiplier * d_model
    print(
        f"capturing input to {layer_path}[{layer_index}] "
        f"({len(layers)} layers, d_model={d_model}); SAE width={d_sae:,}, k={args.k}"
    )
    config = ExperimentConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        context_size=args.context_size,
        layer_index=layer_index,
        layer_path=layer_path,
        residual_location="layer_input",
        d_model=d_model,
        d_sae=d_sae,
        width_multiplier=args.width_multiplier,
        k=args.k,
        learning_rate=args.learning_rate,
        model_batch_size=args.model_batch_size,
        sae_batch_size=args.sae_batch_size,
        seed=args.seed,
        device=str(device),
        model_dtype=args.model_dtype,
    )
    sae = TopKSAE(d_model, d_sae, args.k, device)
    capture = ResidualStreamCapture(layers[layer_index])
    try:
        excluded_token_ids = set(tokenizer.all_special_ids)
        processed, evaluation = train_sae(
            model,
            capture,
            sae,
            train_tokens,
            validation_tokens,
            tokenizer.pad_token_id,
            device,
            args,
            config,
            excluded_token_ids,
        )
        print(f"trained on {processed:,} tokens")

        if evaluation is None:
            evaluation = evaluate_sae(
                model,
                capture,
                sae,
                validation_tokens,
                tokenizer.pad_token_id,
                device,
                args,
                excluded_token_ids,
            )
        validation, fire_counts, max_activation, report_max_activation = evaluation
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(validation, sort_keys=True))

        torch.save(
            {
                "fire_counts": torch.from_numpy(fire_counts),
                "max_activation": torch.from_numpy(max_activation),
                "max_non_special_activation": torch.from_numpy(report_max_activation),
            },
            args.output_dir / "validation_feature_stats.pt",
        )
        if not args.skip_examples and args.report_features > 0:
            eligible = np.flatnonzero(
                (fire_counts >= args.examples_per_feature) & (report_max_activation > 0)
            )
            selected = eligible[
                np.argsort(-report_max_activation[eligible])[: args.report_features]
            ]
            reports = collect_top_examples(
                model,
                capture,
                sae,
                validation_tokens,
                selected,
                tokenizer,
                tokenizer.pad_token_id,
                device,
                args,
                excluded_token_ids,
            )
            reports = rank_reports_by_token_coherence(reports, validation_tokens, tokenizer)
            (args.output_dir / "top_examples.json").write_text(
                json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
            )
            write_example_markdown(args.output_dir / "top_examples.md", reports)
    finally:
        capture.close()


if __name__ == "__main__":
    main()

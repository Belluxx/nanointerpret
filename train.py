"""
Train a Top-K sparse autoencoder on Gemma's midpoint residual stream.

Defaults: Gemma 3 270M, FineWeb-Edu sample-10BT, 100M train tokens,
10M validation tokens, context length 256, 16x expansion, K=32, and AuxK
dead-feature prevention with coefficient 1/32.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import TokenCacheSpec, build_token_cache
from src.experiment import (
    ExperimentConfig,
    ResidualStreamCapture,
    default_aux_k,
    estimate_activation_normalization,
    evaluate_sae,
    find_transformer_layers,
    format_metrics_line,
    train_sae,
)
from src.runtime import choose_device
from src.sae import TopKSAE


MODEL_ID = "google/gemma-3-270m"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--width-multiplier", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--aux-k", type=int, default=None, help="Dead latents used by AuxK. Default: nearest power of two to d_model / 2.")
    parser.add_argument("--aux-k-coef", type=float, default=1 / 32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-batch-size", type=int, default=32, help="Contexts processed together; lower this if memory is limited.")
    parser.add_argument("--sae-batch-size", type=int, default=4096, help="SAE token microbatch; lower this if memory is limited.")
    parser.add_argument("--normalization-tokens", type=int, default=1_000_000, help="Training-token sample used to estimate one global activation scale.")
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000_000, help="Save and evaluate a checkpoint after this many training tokens.")
    parser.add_argument("--dead-window", type=int, default=10_000_000)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation-layer", type=int, default=None, help="Layer whose input is captured. Default: len(transformer.layers) // 2.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sae_gemma_3_270m"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_tokens,
        args.validation_tokens,
        args.context_size,
        args.width_multiplier,
        args.k,
        args.learning_rate,
        args.model_batch_size,
        args.sae_batch_size,
        args.normalization_tokens,
        args.log_every,
        args.checkpoint_every,
        args.dead_window,
    )
    if min(positive) <= 0:
        raise ValueError("counts, sizes, intervals, and learning rate must be positive")
    if args.gradient_clip < 0:
        raise ValueError("gradient clip cannot be negative")
    if args.aux_k is not None and args.aux_k <= 0:
        raise ValueError("aux-k must be positive")
    if args.aux_k_coef < 0:
        raise ValueError("aux-k-coef must be non-negative")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_activation_normalization(
    args: argparse.Namespace,
    model: torch.nn.Module,
    capture: ResidualStreamCapture,
    train_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    d_model: int,
) -> float:
    checkpoint_path = args.output_dir / "checkpoint.pt"
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume: {checkpoint_path} does not exist")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if not isinstance(saved_config, dict):
            raise ValueError("cannot resume: checkpoint has no valid experiment configuration")
        if "activation_scale" not in saved_config:
            raise ValueError(
                "cannot resume this checkpoint because it predates activation normalization"
            )
        activation_scale = float(saved_config["activation_scale"])
        print(f"reusing activation scale {activation_scale:.8g} from {checkpoint_path}")
        return activation_scale

    if checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --resume or choose another --output-dir"
        )
    return estimate_activation_normalization(
        model,
        capture,
        train_tokens,
        pad_token_id,
        device,
        args,
        d_model,
    )


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

    cache_spec = TokenCacheSpec(
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
    )
    train_path, validation_path = build_token_cache(tokenizer, cache_spec)
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
    aux_k = default_aux_k(d_model) if args.aux_k is None else args.aux_k
    if aux_k > d_sae:
        raise ValueError(f"aux-k must be at most the SAE width ({d_sae}), got {aux_k}")
    print(
        f"capturing input to {layer_path}[{layer_index}] "
        f"({len(layers)} layers, d_model={d_model}); SAE width={d_sae:,}, "
        f"k={args.k}, AuxK={aux_k}"
    )
    with ResidualStreamCapture(layers[layer_index]) as capture:
        activation_scale = prepare_activation_normalization(
            args,
            model,
            capture,
            train_tokens,
            tokenizer.pad_token_id,
            device,
            d_model,
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
            aux_k=aux_k,
            aux_k_coef=args.aux_k_coef,
            dead_window=args.dead_window,
            learning_rate=args.learning_rate,
            model_batch_size=args.model_batch_size,
            sae_batch_size=args.sae_batch_size,
            normalization_tokens=args.normalization_tokens,
            activation_scale=activation_scale,
            seed=args.seed,
            device=str(device),
            model_dtype=args.model_dtype,
        )
        sae = TopKSAE(d_model, d_sae, args.k, device)
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
        )
        print(f"trained on {processed:,} tokens")

        from src.plot import save_feature_density_plot, save_training_plot

        save_training_plot(
            args.output_dir / "train_metrics.jsonl",
            args.output_dir / "training_metrics.png",
        )
        save_feature_density_plot(
            args.output_dir / "checkpoint_metrics.jsonl",
            args.output_dir / "validation_feature_density.png",
        )

        if evaluation is None:
            evaluation = evaluate_sae(
                model,
                capture,
                sae,
                validation_tokens,
                tokenizer.pad_token_id,
                device,
                config,
            )
        validation = evaluation
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        print(format_metrics_line(validation))


if __name__ == "__main__":
    main()

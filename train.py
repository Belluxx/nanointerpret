from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    RESIDUAL_DTYPE,
    ResidualCacheSpec,
    TokenCacheSpec,
    build_token_cache,
    iter_residual_batches,
    load_residual_cache_metadata,
    residual_cache_paths,
)
from src.experiment import (
    ExperimentConfig,
    ResidualBatchFactory,
    ResidualStreamCapture,
    capture_residual_cache,
    default_aux_k,
    estimate_activation_normalization,
    evaluate_sae,
    find_transformer_layers,
    format_metrics_line,
    iter_captured_residual_batches,
    train_sae,
)
from src.runtime import ATTENTION_IMPLEMENTATION, choose_device
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
    parser.add_argument("--model-batch-size", type=int, default=128, help="Contexts processed together; lower this if memory is limited.")
    parser.add_argument("--sae-batch-size", type=int, default=8192, help="SAE token microbatch; lower this if memory is limited.")
    parser.add_argument("--normalization-tokens", type=int, default=1_000_000, help="Training-token sample used to estimate one global activation scale.")
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000_000, help="Save and evaluate a checkpoint after this many training tokens.")
    parser.add_argument("--dead-window", type=int, default=10_000_000)
    parser.add_argument("--gradient-clip", type=float, default=None, metavar="MAX_NORM", help="Enable gradient clipping with the specified positive maximum norm.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation-layer", type=int, default=None, help="Layer whose input is captured. Default: len(transformer.layers) // 2.")
    parser.add_argument("--no-subtract-pre-bias", action="store_false", dest="subtract_pre_bias", help="Do not subtract the learned decoder bias from activations before encoding.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sae_gemma_3_270m"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"))
    parser.add_argument("--residual-cache-dir", type=Path, default=Path("artifacts/residual_cache"))
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cache-activations", action="store_true", help="Cache residual activations before training instead of streaming them.")
    mode.add_argument("--cache-only", action="store_true", help="Capture the residual cache, then exit before SAE training.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_capture_inputs(
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_spec = TokenCacheSpec(
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
    )
    train_path, validation_path = build_token_cache(tokenizer, token_spec)
    train_tokens = np.memmap(train_path, mode="r", dtype=np.uint32)
    validation_tokens = np.memmap(validation_path, mode="r", dtype=np.uint32)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(device)
    model.eval().requires_grad_(False)
    layer_path, layers = find_transformer_layers(model)
    layer_index = (
        len(layers) // 2 if args.activation_layer is None else args.activation_layer
    )
    if not 0 <= layer_index < len(layers):
        raise ValueError(
            f"activation layer must be in [0, {len(layers) - 1}], got {layer_index}"
        )
    metadata = {
        "layer_index": layer_index,
        "layer_path": layer_path,
        "layer_count": len(layers),
        "residual_location": "layer_input",
        "d_model": int(model.config.hidden_size),
    }
    return tokenizer, model, layers[layer_index], train_tokens, validation_tokens, metadata


def build_residual_cache(
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    spec: ResidualCacheSpec,
    cache_paths: tuple[Path, Path, Path],
) -> None:
    tokenizer, model, layer, train_tokens, validation_tokens, layer_metadata = (
        load_capture_inputs(args, device, model_dtype)
    )
    metadata = asdict(spec)
    metadata.pop("cache_dir")
    metadata.update(layer_metadata)

    with ResidualStreamCapture(layer) as capture:
        capture_residual_cache(
            model,
            capture,
            train_tokens,
            validation_tokens,
            tokenizer.pad_token_id,
            device,
            args,
            cache_paths,
            metadata,
        )


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    metadata: dict,
    train_batches: ResidualBatchFactory,
    validation_batches: ResidualBatchFactory,
) -> None:
    d_model = int(metadata["d_model"])
    d_sae = args.width_multiplier * d_model
    aux_k = default_aux_k(d_model) if args.aux_k is None else args.aux_k
    print(
        f"Device: {device} | Mode: {'cached' if args.cache_activations else 'streaming'} | "
        f"Model batch: {args.model_batch_size} | SAE batch: {args.sae_batch_size} | "
        f"Layer: {metadata['layer_index']} | Model width: {d_model} | "
        f"SAE width: {d_sae:,} | k: {args.k} | "
        f"AuxK: {'off' if args.aux_k_coef == 0 else aux_k}"
    )

    if args.normalization_tokens <= 0:
        raise ValueError("--normalization-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    if args.resume:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        activation_scale = float(checkpoint["config"]["activation_scale"])
        initial_pre_bias = None
        print(
            f"reusing activation scale {activation_scale:.8g} "
            f"from {checkpoint_path}"
        )
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists; pass --resume or choose "
                "another --output-dir"
            )
        activation_scale, initial_pre_bias = estimate_activation_normalization(
            train_batches,
            args.train_tokens,
            d_model,
            device,
            args,
        )
    config = ExperimentConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        context_size=args.context_size,
        layer_index=int(metadata["layer_index"]),
        width_multiplier=args.width_multiplier,
        k=args.k,
        aux_k=aux_k,
        aux_k_coef=args.aux_k_coef,
        dead_window=args.dead_window,
        learning_rate=args.learning_rate,
        gradient_clip=args.gradient_clip,
        model_batch_size=args.model_batch_size,
        sae_batch_size=args.sae_batch_size,
        normalization_tokens=args.normalization_tokens,
        activation_scale=activation_scale,
        subtract_pre_bias=args.subtract_pre_bias,
        seed=args.seed,
        model_dtype=args.model_dtype,
    )
    sae = TopKSAE(
        d_model,
        d_sae,
        args.k,
        device,
        subtract_pre_bias=args.subtract_pre_bias,
    )
    if initial_pre_bias is not None:
        with torch.no_grad():
            sae.decoder_bias.copy_(initial_pre_bias)
    _, evaluation = train_sae(
        sae,
        train_batches,
        validation_batches,
        device,
        args,
        config,
    )
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
            sae,
            validation_batches,
            device,
            config,
        )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
    )
    print(format_metrics_line(evaluation))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    model_dtype = getattr(torch, args.model_dtype)

    if not args.cache_activations and not args.cache_only:
        tokenizer, model, layer, train_data, validation_data, metadata = (
            load_capture_inputs(args, device, model_dtype)
        )
        with ResidualStreamCapture(layer) as capture:
            batch_function = partial(
                iter_captured_residual_batches,
                model,
                capture,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                context_size=args.context_size,
                model_batch_size=args.model_batch_size,
                seed=args.seed,
            )
            run_training(
                args,
                device,
                metadata,
                partial(batch_function, train_data, shuffle=True),
                partial(batch_function, validation_data, shuffle=False),
            )
        return

    spec = ResidualCacheSpec(
        cache_dir=args.residual_cache_dir,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        context_size=args.context_size,
        activation_layer=args.activation_layer,
        model_dtype=args.model_dtype,
    )
    cache_paths = residual_cache_paths(spec)
    train_path, validation_path, _ = cache_paths
    metadata = load_residual_cache_metadata(spec)
    if metadata is None:
        build_residual_cache(args, device, model_dtype, spec, cache_paths)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        metadata = load_residual_cache_metadata(spec)
        if metadata is None:
            raise RuntimeError("the residual cache failed validation after capture")
    if args.cache_only:
        print(f"Residual cache: {args.residual_cache_dir}")
        return

    d_model = int(metadata["d_model"])
    train_data = np.memmap(
        train_path,
        mode="r",
        dtype=RESIDUAL_DTYPE,
        shape=(args.train_tokens, d_model),
    )
    validation_data = np.memmap(
        validation_path,
        mode="r",
        dtype=RESIDUAL_DTYPE,
        shape=(args.validation_tokens, d_model),
    )
    batch_function = partial(
        iter_residual_batches,
        batch_size=args.context_size * args.model_batch_size,
        seed=args.seed,
    )
    run_training(
        args,
        device,
        metadata,
        partial(batch_function, train_data, shuffle=True),
        partial(batch_function, validation_data, shuffle=False),
    )


if __name__ == "__main__":
    main()

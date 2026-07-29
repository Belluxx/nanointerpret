from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    RESIDUAL_DTYPE,
    ResidualCacheSpec,
    TokenCacheSpec,
    build_token_cache,
    load_residual_cache_metadata,
    residual_cache_paths,
)
from src.experiment import (
    ExperimentConfig,
    ResidualStreamCapture,
    capture_residual_cache,
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
    parser.add_argument("--cache-only", action="store_true", help="Capture the residual cache, then exit before SAE training.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_residual_cache(
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    spec: ResidualCacheSpec,
    cache_paths: tuple[Path, Path, Path],
) -> None:
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
        attn_implementation="eager",
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

    d_model = int(model.config.hidden_size)
    metadata = asdict(spec)
    metadata.pop("cache_dir")
    metadata.update(
        {
            "layer_index": layer_index,
            "layer_path": layer_path,
            "layer_count": len(layers),
            "residual_location": "layer_input",
            "d_model": d_model,
        }
    )
    print(
        f"capturing input to {layer_path}[{layer_index}] "
        f"({len(layers)} layers, d_model={d_model})"
    )
    with ResidualStreamCapture(layers[layer_index]) as capture:
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    model_dtype = getattr(torch, args.model_dtype)

    print(
        f"device={device}, model dtype={model_dtype}, "
        f"model batch={args.model_batch_size}, "
        f"SAE batch={args.sae_batch_size}"
    )
    residual_cache_spec = ResidualCacheSpec(
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
    cache_paths = residual_cache_paths(residual_cache_spec)
    residual_train_path, residual_validation_path, _ = cache_paths
    cache_metadata = load_residual_cache_metadata(residual_cache_spec)
    if cache_metadata is None:
        print("phase 1/2: capturing residual cache")
        build_residual_cache(
            args, device, model_dtype, residual_cache_spec, cache_paths
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        cache_metadata = load_residual_cache_metadata(residual_cache_spec)
        if cache_metadata is None:
            raise RuntimeError("the residual cache failed validation after capture")
    else:
        print(f"using residual cache at {args.residual_cache_dir}")

    print(f"cached train residuals: {residual_train_path}")
    print(f"cached validation residuals: {residual_validation_path}")
    if args.cache_only:
        return
    if args.normalization_tokens <= 0:
        raise ValueError("--normalization-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("phase 2/2: training SAE from cached residuals (LLM not loaded)")
    d_model = int(cache_metadata["d_model"])
    train_residuals = np.memmap(
        residual_train_path,
        mode="r",
        dtype=RESIDUAL_DTYPE,
        shape=(args.train_tokens, d_model),
    )
    validation_residuals = np.memmap(
        residual_validation_path,
        mode="r",
        dtype=RESIDUAL_DTYPE,
        shape=(args.validation_tokens, d_model),
    )

    d_sae = args.width_multiplier * d_model
    aux_k = default_aux_k(d_model) if args.aux_k is None else args.aux_k
    print(
        f"cached input to {cache_metadata['layer_path']}"
        f"[{cache_metadata['layer_index']}] (d_model={d_model}); "
        f"SAE width={d_sae:,}, "
        f"k={args.k}, AuxK={'off' if args.aux_k_coef == 0 else aux_k}, "
        f"pre-bias subtraction={'on' if args.subtract_pre_bias else 'off'}"
    )
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
            train_residuals,
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
        layer_index=int(cache_metadata["layer_index"]),
        layer_path=str(cache_metadata["layer_path"]),
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
        subtract_pre_bias=args.subtract_pre_bias,
        seed=args.seed,
        device=str(device),
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
    processed, evaluation = train_sae(
        sae,
        train_residuals,
        validation_residuals,
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
            sae,
            validation_residuals,
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

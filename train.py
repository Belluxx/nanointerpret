from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer

from src.data import (
    ResidualCacheSpec,
    TokenCacheSpec,
    build_token_cache,
    iter_residual_batches,
    load_residual_cache_metadata,
    residual_cache_paths,
)
from src.experiment import (
    ExperimentConfig,
    calibrate_activations,
    capture_residual_cache,
    default_aux_k,
    evaluate_downstream_kl,
    iter_captured_residual_batches,
    train_sae,
)
from src.misc import experiment_output_dir
from src.runtime import (
    choose_device,
    compile_transformer_prefix,
    find_transformer_layers,
    load_causal_lm,
)
from src.sae import TopKSAE

MODEL_ID = "google/gemma-3-270m"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID, help=f"Hugging Face causal language model to analyze. Default: {MODEL_ID}.")
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16", help="Language-model inference dtype used while capturing residuals. Default: bfloat16.")
    parser.add_argument("--activation-layer", type=int, default=None, help="Layer whose input is captured. Default: len(transformer.layers) // 2.")
    parser.add_argument("--no-compile-model", action="store_false", dest="compile_model", help="Disable compilation of transformer layers before the capture point on MPS.")
    parser.add_argument("--dataset-id", default=DATASET_ID, help=f"Hugging Face text dataset used to build the token splits. Default: {DATASET_ID}.")
    parser.add_argument("--dataset-config", default=DATASET_CONFIG, help=f"Named Hugging Face dataset configuration or subset. Default: {DATASET_CONFIG}.")
    parser.add_argument("--context-size", type=int, default=256, help="Tokens in each language-model input context. Default: 256.")
    parser.add_argument("--train-tokens", type=int, default=100_000_000, help="Number of dataset tokens used to train the SAE. Default: 100000000.")
    parser.add_argument("--validation-tokens", type=int, default=10_000_000, help="Dedicated token split used to evaluate the SAE. Default: 10000000.")
    parser.add_argument("--recording-tokens", type=int, default=10_000_000, help="Dedicated token split for recording feature activations. Default: 10000000.")
    parser.add_argument("--model-batch-size", type=int, default=32, help="Contexts processed together; lower this if memory is limited.")
    parser.add_argument("--normalization-tokens", type=int, default=1_000_000, help="Training-token sample used to estimate one global activation scale.")
    parser.add_argument("--no-passthrough-massive-dims", action="store_false", dest="passthrough_massive_dims", help="Let the SAE model every residual dimension, including massive-activation dims whose std is far above the rest.")
    parser.add_argument("--width-multiplier", type=int, default=16, help="SAE feature count as a multiple of the model residual width. Default: 16.")
    parser.add_argument("--k", type=int, default=16, help="Maximum number of SAE features active for each token. Default: 16.")
    parser.add_argument("--aux-k", type=int, default=None, help="Dead latents used by AuxK. Default: nearest power of two to d_model / 2.")
    parser.add_argument("--aux-k-coef", type=float, default=1 / 32, help="Weight of the AuxK reconstruction loss; set to 0 to disable AuxK. Default: 1/32.")
    parser.add_argument("--no-subtract-pre-bias", action="store_false", dest="subtract_pre_bias", help="Do not subtract the learned decoder bias from activations before encoding.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate. Default: 3e-4 * sqrt(32768 / SAE feature count).")
    parser.add_argument("--sae-batch-size", type=int, default=4096, help="SAE token microbatch; lower this if memory is limited.")
    parser.add_argument("--gradient-clip", type=float, default=None, metavar="MAX_NORM", help="Enable gradient clipping with the specified positive maximum norm.")
    parser.add_argument("--dead-window", type=int, default=10_000_000, help="Tokens a feature may go without firing before AuxK treats it as dead. Default: 10000000.")
    parser.add_argument("--log-every", type=int, default=100_000, help="Training-token interval between metric records. Default: 100000.")
    parser.add_argument("--checkpoint-every", type=int, default=50_000_000, help="Save a checkpoint after this many training tokens.")
    parser.add_argument("--validate-every", type=int, default=50_000_000, help="Validate after this many training tokens.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"), help="Directory for reusable tokenized dataset splits. Default: artifacts/token_cache.")
    parser.add_argument("--residual-cache-dir", type=Path, default=Path("artifacts/residual_cache"), help="Directory for fp16 residual activations saved by cached modes. Default: artifacts/residual_cache.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cache-activations", action="store_true", help="Cache residual activations before training instead of streaming them.")
    mode.add_argument("--cache-only", action="store_true", help="Capture the residual cache, then exit before SAE training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output directory. Default: generated automatically under artifacts/.",)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    d_model: int,
    layer_index: int,
    train_batches,
    validation_batches,
    downstream_kl_evaluator,
) -> None:
    d_sae = args.width_multiplier * d_model
    aux_k = default_aux_k(d_model) if args.aux_k is None else args.aux_k

    if args.learning_rate is None:
        args.learning_rate = 3e-4 * math.sqrt(32768 / d_sae)

    if args.output_dir is None:
        args.output_dir = experiment_output_dir(
            args.model_id,
            layer_index,
            args.width_multiplier,
            args.k,
            args.train_tokens,
        )
    mode = "cached" if args.cache_activations else "streaming"
    print(
        f"Device: {device} | Mode: {mode} | "
        f"Model batch: {args.model_batch_size} | SAE batch: {args.sae_batch_size} | "
        f"Layer: {layer_index} | Model width: {d_model} | "
        f"SAE width: {d_sae:,} | k: {args.k} | "
        f"AuxK: {'off' if args.aux_k_coef == 0 else aux_k}"
    )
    print(f"Output: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    final_path = args.output_dir / "sae_final.pt"
    if args.resume:
        config_path = args.output_dir / "config.json"
        saved_config = json.loads(config_path.read_text())
        activation_scale = float(saved_config["activation_scale"])
        passthrough_dims = saved_config["passthrough_dims"]
        initial_pre_bias = None
        print(
            f"reusing activation scale {activation_scale:.8g} "
            f"from {config_path}"
        )
    else:
        if final_path.exists():
            raise FileExistsError(
                f"{final_path} already exists; choose another --output-dir"
            )
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists; pass --resume or choose another "
                "--output-dir"
            )
        passthrough_dims, activation_scale, initial_pre_bias = calibrate_activations(
            train_batches,
            args.train_tokens,
            device,
            args.normalization_tokens,
            args.subtract_pre_bias,
            args.passthrough_massive_dims,
        )
    config = ExperimentConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        recording_tokens=args.recording_tokens,
        context_size=args.context_size,
        layer_index=layer_index,
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
        cache_activations=args.cache_activations,
        passthrough_dims=passthrough_dims,
    )
    sae = TopKSAE(
        d_model - len(passthrough_dims),
        d_sae,
        args.k,
        device,
        subtract_pre_bias=args.subtract_pre_bias,
    )
    if initial_pre_bias is not None:
        with torch.no_grad():
            sae.decoder_bias.copy_(initial_pre_bias)
    evaluation = train_sae(
        sae,
        train_batches,
        validation_batches,
        device,
        config,
        args.output_dir,
        args.resume,
        args.log_every,
        args.checkpoint_every,
        args.validate_every,
        downstream_kl_evaluator,
    )
    from src.plot import save_feature_density_plot, save_training_plot

    save_training_plot(
        args.output_dir / "train_metrics.jsonl",
        args.output_dir / "evaluation_metrics.jsonl",
        args.output_dir / "training_metrics.png",
    )
    save_feature_density_plot(
        args.output_dir / "evaluation_metrics.jsonl",
        args.output_dir / "validation_feature_density.png",
    )

    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model_dtype = getattr(torch, args.model_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    token_spec = TokenCacheSpec(
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        recording_tokens=args.recording_tokens,
    )
    train_path, validation_path = build_token_cache(tokenizer, token_spec)
    train_tokens = np.memmap(train_path, mode="r", dtype=np.uint32)
    validation_tokens = np.memmap(validation_path, mode="r", dtype=np.uint32)

    model = load_causal_lm(args.model_id, model_dtype, device)
    layers = find_transformer_layers(model)
    layer_index = len(layers) // 2 if args.activation_layer is None else args.activation_layer
    layer = layers[layer_index]
    d_model = int(model.config.hidden_size)
    if args.compile_model and device.type == "mps":
        compile_transformer_prefix(layers, layer_index)
        print(f"Compiling {layer_index} transformer layers for MPS")

    if args.cache_activations or args.cache_only:
        spec = ResidualCacheSpec(
            cache_dir=args.residual_cache_dir,
            model_id=args.model_id,
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            train_tokens=args.train_tokens,
            validation_tokens=args.validation_tokens,
            context_size=args.context_size,
            layer_index=layer_index,
            model_dtype=args.model_dtype,
        )
        cache_paths = residual_cache_paths(spec)
        if load_residual_cache_metadata(spec) is None:
            if any(path.exists() for path in cache_paths):
                raise RuntimeError("Corrupted residual cache, delete it and rerun to rebuild")
            metadata = asdict(spec)
            metadata.pop("cache_dir")
            metadata["d_model"] = d_model
            capture_residual_cache(
                model,
                layer,
                train_tokens,
                validation_tokens,
                device,
                args.context_size,
                args.model_batch_size,
                cache_paths,
                metadata,
            )
        if args.cache_only:
            print(f"Residual cache: {args.residual_cache_dir}")
            return
        batches = partial(
            iter_residual_batches,
            batch_size=args.context_size * args.model_batch_size,
            seed=args.seed,
        )
        train_data = np.load(cache_paths[0], mmap_mode="r")
        validation_data = np.load(cache_paths[1], mmap_mode="r")
    else:
        batches = partial(
            iter_captured_residual_batches,
            model,
            layer,
            device=device,
            context_size=args.context_size,
            model_batch_size=args.model_batch_size,
            seed=args.seed,
        )
        train_data, validation_data = train_tokens, validation_tokens

    def downstream_kl_evaluator(sae: TopKSAE, config: ExperimentConfig) -> dict:
        return evaluate_downstream_kl(
            sae,
            model,
            layer,
            validation_tokens,
            device,
            args.context_size,
            config.activation_scale,
            config.passthrough_dims,
        )

    run_training(
        args,
        device,
        d_model,
        layer_index,
        partial(batches, train_data, shuffle=True),
        partial(batches, validation_data, shuffle=False),
        downstream_kl_evaluator,
    )


if __name__ == "__main__":
    main()

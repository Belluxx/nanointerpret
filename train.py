"""Train a Top-K SAE on a language model's residual stream."""

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.data import MIXES, cache_name, context_batches, context_splits, read_residual_cache, residual_cache, shuffled_stages, token_cache, train_stages
from src.experiment import Config, build_sae, calibrate, downstream_kl, evaluate_sae, train_sae
from src.plot import save_plots
from src.runtime import capture_layer_input, choose_device, compile_layers_before, find_transformer_layers, load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Hugging Face causal language model to analyze.")
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16", help="Language-model inference dtype used while capturing residuals. Default: bfloat16.")
    parser.add_argument("--activation-layer", type=int, default=None, help="Layer whose input is captured. Default: len(transformer.layers) // 2.")
    parser.add_argument("--no-compile-model", action="store_false", dest="compile_model", help="Disable compilation of transformer layers before the capture point on MPS.")
    parser.add_argument("--mix", choices=MIXES, default="broad", help="Data mix the token splits are drawn from, defined in src/data.py: broad (web, PDFs, code, math, and five languages) or fineweb (FineWeb only). Default: broad.")
    parser.add_argument("--context-size", type=int, default=256, help="Tokens in each language-model input context. Default: 256.")
    parser.add_argument("--train-tokens", type=int, default=100_000_000, help="Number of dataset tokens used to train the SAE. Default: 100000000.")
    parser.add_argument("--validation-tokens", type=int, default=10_000_000, help="Dedicated token split used to evaluate the SAE. Default: 10000000.")
    parser.add_argument("--recording-tokens", type=int, default=10_000_000, help="Dedicated token split for recording feature activations. Default: 10000000.")
    parser.add_argument("--model-batch-size", type=int, default=32, help="Contexts processed together; lower this if memory is limited.")
    parser.add_argument("--normalization-tokens", type=int, default=1_000_000, help="Training-token sample used to calibrate sink tokens, pass-through dims, and the activation scale.")
    parser.add_argument("--no-exclude-sinks", action="store_false", dest="exclude_sinks", help="Let the SAE model attention-sink tokens: the first token of each context and those with at least half its norm, when it stands out.")
    parser.add_argument("--no-passthrough-massive-dims", action="store_false", dest="passthrough_massive_dims", help="Let the SAE model every residual dimension, including massive-activation dims whose std is far above the rest.")
    parser.add_argument("--width-multiplier", type=int, default=16, help="SAE feature count as a multiple of the model residual width. Default: 16.")
    parser.add_argument("--k", type=int, default=32, help="Maximum number of SAE features active for each token. Default: 32.")
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
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"), help="Directory for reusable tokenized datasets. Default: artifacts/token_cache.")
    parser.add_argument("--residual-cache-dir", type=Path, default=Path("artifacts/residual_cache"), help="Directory for fp16 residual activations saved by cached modes. Default: artifacts/residual_cache.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cache-activations", action="store_true", help="Cache residual activations before training instead of streaming them.")
    mode.add_argument("--cache-only", action="store_true", help="Capture the residual cache, then exit before SAE training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output directory. Default: generated automatically under artifacts/.")
    parser.add_argument("--resume", action="store_true", help="Continue from the output directory's checkpoint, with the original arguments; a larger --train-tokens extends the run.")
    return parser.parse_args()


def compact_count(count: int) -> str:
    for divisor, suffix in ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k")):
        if count % divisor == 0:
            return f"{count // divisor}{suffix}"
    return str(count)


def extension_history(saved: dict | None, args: argparse.Namespace) -> list[int]:
    # The run's earlier `train_tokens`; a resumed run with a larger --train-tokens adds its saved one.
    if saved is None:
        return []
    if args.train_tokens == saved["train_tokens"]:
        return saved["extended_from"]
    if args.train_tokens // args.context_size <= saved["train_tokens"] // args.context_size:
        raise ValueError(f"--resume can only grow --train-tokens, by at least one context, from {saved['train_tokens']:,}")
    print(f"Extending training from {saved['train_tokens']:,} to {args.train_tokens:,} tokens")
    return [*saved["extended_from"], saved["train_tokens"]]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    # Tokenizing streams datasets through memory, so it runs before the model is loaded.
    splits = context_splits(args.train_tokens, args.validation_tokens, args.recording_tokens, args.context_size)
    all_contexts, all_sources = token_cache(
        args.cache_dir, args.model_id, args.mix, args.context_size, splits["train"].stop
    )
    contexts = {split: all_contexts[splits[split]] for split in ("train", "validation")}

    model = load_causal_lm(args.model_id, args.model_dtype, device)
    layers = find_transformer_layers(model)
    layer_index = len(layers) // 2 if args.activation_layer is None else args.activation_layer
    layer = layers[layer_index]
    d_model = model.config.hidden_size
    d_sae = args.width_multiplier * d_model
    aux_k = args.aux_k or 1 << round(math.log2(d_model / 2))
    output_dir = args.output_dir or Path("artifacts") / (
        f"{args.model_id.split('/')[-1].lower()}_l{layer_index}_w{args.width_multiplier}_k{args.k}_"
        f"{compact_count(args.train_tokens)}"
    )
    config_path = output_dir / "config.json"
    if args.resume and not config_path.exists():
        raise FileNotFoundError(f"{output_dir} holds no run to resume; pass the run's --output-dir")
    if not args.resume and any(output_dir.glob("*.pt")):
        raise FileExistsError(f"{output_dir} already holds a run; pass --resume or choose another --output-dir")
    saved = json.loads(config_path.read_text()) if args.resume else None
    extended_from = extension_history(saved, args)
    stages = train_stages([*extended_from, args.train_tokens], args.context_size)
    if args.compile_model and device.type == "mps":
        compile_layers_before(layers, layer_index)

    def capture(tokens: np.ndarray) -> torch.Tensor:
        return capture_layer_input(model, layer, torch.from_numpy(tokens.astype(np.int64)).to(device))

    def range_cache(split: str, contexts_range: slice) -> np.ndarray:
        start, stop = splits[split].start + contexts_range.start, splits[split].start + contexts_range.stop
        name = cache_name(
            args.model_id, args.mix, f"contexts{start}-{stop}", f"ctx{args.context_size}", f"layer{layer_index}", args.model_dtype
        )
        return residual_cache(
            args.residual_cache_dir / f"{name}.npy", contexts[split][contexts_range], capture, args.model_batch_size, d_model
        )

    caches = {}
    if args.cache_activations or args.cache_only:
        caches = {
            "train": [range_cache("train", stage) for stage in stages],
            "validation": [range_cache("validation", slice(0, len(contexts["validation"])))],
        }
        if args.cache_only:
            print(f"Residual cache: {args.residual_cache_dir}")
            return

    def batches(split: str, skip: int = 0) -> Iterator[torch.Tensor]:
        # Training contexts come shuffled, validation ones in order.
        order = shuffled_stages(stages, args.seed) if split == "train" else np.arange(len(contexts[split]))
        ids = context_batches(order, args.model_batch_size, skip)
        if caches:
            return read_residual_cache(caches[split], ids, device)
        return (capture(contexts[split][batch_ids]).flatten(0, 1) for batch_ids in ids)

    print(
        f"Device: {device} | Mode: {'cached' if caches else 'streaming'} | "
        f"Model batch: {args.model_batch_size} | SAE batch: {args.sae_batch_size} | "
        f"Layer: {layer_index} | Model width: {d_model} | SAE width: {d_sae:,} | k: {args.k} | "
        f"AuxK: {'off' if args.aux_k_coef == 0 else aux_k}"
    )
    print(f"Output: {output_dir}")

    if saved:
        sink_norm_threshold = saved["sink_norm_threshold"]
        passthrough_dims, activation_scale = saved["passthrough_dims"], saved["activation_scale"]
    else:
        sink_norm_threshold, passthrough_dims, activation_scale = calibrate(
            batches("train"), args.normalization_tokens, args.context_size, args.exclude_sinks, args.passthrough_massive_dims
        )
    config = Config(
        model_id=args.model_id,
        model_dtype=args.model_dtype,
        layer_index=layer_index,
        d_model=d_model,
        mix=args.mix,
        context_size=args.context_size,
        train_tokens=args.train_tokens,
        extended_from=extended_from,
        validation_tokens=args.validation_tokens,
        recording_tokens=args.recording_tokens,
        width_multiplier=args.width_multiplier,
        k=args.k,
        aux_k=aux_k,
        aux_k_coef=args.aux_k_coef,
        subtract_pre_bias=args.subtract_pre_bias,
        learning_rate=args.learning_rate or 3e-4 * math.sqrt(32768 / d_sae),
        gradient_clip=args.gradient_clip,
        dead_window=args.dead_window,
        model_batch_size=args.model_batch_size,
        sae_batch_size=args.sae_batch_size,
        normalization_tokens=args.normalization_tokens,
        sink_norm_threshold=sink_norm_threshold,
        passthrough_dims=passthrough_dims,
        activation_scale=activation_scale,
        seed=args.seed,
    )
    if saved and asdict(replace(config, train_tokens=saved["train_tokens"], extended_from=saved["extended_from"])) != saved:
        raise ValueError("--resume needs the arguments of the original run, apart from a larger --train-tokens")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")

    sae = build_sae(config).to(device)
    if config.subtract_pre_bias and not args.resume:
        sae.init_pre_bias(next(batches("train")))

    def evaluate() -> dict:
        # Full-vocabulary logits dwarf residuals, so the KL runs on an eighth of the model batch.
        return {
            **evaluate_sae(sae, batches("validation"), contexts["validation"].size, config.sae_batch_size),
            **downstream_kl(
                sae, model, layer, contexts["validation"], all_sources[splits["validation"]],
                [source.name for source in MIXES[config.mix]], max(1, config.model_batch_size // 8),
            ),
        }

    train_sae(
        sae,
        config,
        lambda skip: batches("train", skip=skip),
        evaluate,
        output_dir,
        args.resume,
        args.log_every,
        args.checkpoint_every,
        args.validate_every,
    )
    save_plots(output_dir)


if __name__ == "__main__":
    main()

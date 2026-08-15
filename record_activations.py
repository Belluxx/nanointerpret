from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.data import (
    ACTIVATION_VALUE_DTYPE,
    TokenCacheSpec,
    iter_context_batches,
    save_activations,
    token_cache_is_valid,
    token_cache_paths,
)
from src.experiment import ResidualStreamCapture, find_transformer_layers
from src.runtime import choose_device, load_causal_lm, load_tokenizer
from src.sae import FIRING_THRESHOLD, TopKSAE, load_sae


CACHE_DIR = Path("artifacts/token_cache")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record SAE feature activations on the local evaluation set."
    )
    parser.add_argument(
        "--sae-dir",
        type=Path,
        required=True,
        help="Training output directory containing config.json and sae_final.pt.",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <sae-dir>/activations.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="Process at most this many evaluation tokens.",
    )
    parser.add_argument(
        "--model-batch-size",
        type=positive_int,
        default=None,
        help="Contexts processed together. Default: the training configuration.",
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    return parser.parse_args()


def load_config(sae_dir: Path) -> dict:
    config_path = sae_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"SAE configuration not found: {config_path}")
    return json.loads(config_path.read_text())


def evaluation_cache(config: dict, cache_dir: Path) -> tuple[Path, int] | None:
    spec = TokenCacheSpec(
        cache_dir=cache_dir,
        model_id=config["model_id"],
        dataset_id=config["dataset_id"],
        dataset_config=config["dataset_config"],
        train_tokens=int(config["train_tokens"]),
        validation_tokens=int(config["validation_tokens"]),
    )
    _, validation_path, _ = token_cache_paths(spec)
    if spec.validation_tokens <= 0 or not token_cache_is_valid(
        spec, validation_only=True
    ):
        return None
    return validation_path, spec.validation_tokens


def encode_activations(
    sae: TopKSAE,
    residuals: Tensor,
    activation_scale: float,
    batch_size: int,
    feature_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = []
    values = []
    for start in range(0, len(residuals), batch_size):
        x = residuals[start : start + batch_size].float()
        x.mul_(activation_scale)
        batch_indices, batch_values = sae.encode(x)
        indices.append(batch_indices.cpu())
        values.append(batch_values.cpu())
    indices = torch.cat(indices)
    values = torch.cat(values)
    firing = values > FIRING_THRESHOLD
    counts = firing.sum(dim=1).numpy().astype(np.uint32, copy=False)
    feature_ids = indices[firing].numpy().astype(feature_dtype, copy=False)
    active_values = values[firing].numpy().astype(
        ACTIVATION_VALUE_DTYPE, copy=False
    )
    return counts, feature_ids, active_values


@torch.inference_mode()
def write_activations(
    output_path: Path,
    pad_token_id: int,
    model,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    evaluation_tokens: np.ndarray,
    config: dict,
    device: torch.device,
    model_batch_size: int,
) -> None:
    context_size = int(config["context_size"])
    token_count = len(evaluation_tokens)
    metadata = {
        "model_id": config["model_id"],
        "context_size": context_size,
        "layer_index": int(config["layer_index"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [
        output_path.with_name(output_path.name + suffix)
        for suffix in (".feature_ids.tmp", ".values.tmp", ".row_ptr.tmp")
    ]
    feature_temporary, values_temporary, row_ptr_temporary = temporary_paths
    pointer_dtype = (
        np.uint32
        if token_count * sae.k <= np.iinfo(np.uint32).max
        else np.uint64
    )
    feature_dtype = np.min_scalar_type(sae.d_sae - 1)
    row_ptr = None
    try:
        row_ptr = np.memmap(
            row_ptr_temporary,
            mode="w+",
            dtype=pointer_dtype,
            shape=(token_count + 1,),
        )
        row_ptr[0] = 0
        processed_tokens = 0
        active_count = 0
        feature_counts = np.zeros(sae.d_sae, dtype=np.uint64)
        feature_max = np.zeros(sae.d_sae, dtype=np.float32)
        with feature_temporary.open("wb") as feature_output, values_temporary.open(
            "wb"
        ) as values_output, tqdm(
            total=token_count,
            unit="tok",
            desc="Record",
            dynamic_ncols=True,
        ) as progress:
            batches = iter_context_batches(
                evaluation_tokens,
                context_size,
                model_batch_size,
                pad_token_id,
                shuffle=False,
                seed=0,
            )
            for input_ids, attention_mask in batches:
                device_input_ids = input_ids.to(device, non_blocking=True)
                device_attention_mask = attention_mask.to(device, non_blocking=True)
                residuals = capture(
                    model, device_input_ids, device_attention_mask
                )[device_attention_mask.bool()]
                counts, feature_ids, active_values = encode_activations(
                    sae,
                    residuals,
                    float(config["activation_scale"]),
                    int(config["sae_batch_size"]),
                    feature_dtype,
                )
                batch_tokens = len(counts)
                cumulative = np.cumsum(counts, dtype=pointer_dtype)
                row_ptr[
                    processed_tokens + 1 : processed_tokens + batch_tokens + 1
                ] = active_count + cumulative
                feature_ids.tofile(feature_output)
                active_values.tofile(values_output)
                feature_counts += np.bincount(
                    feature_ids, minlength=len(feature_counts)
                ).astype(np.uint64)
                np.maximum.at(feature_max, feature_ids, active_values)
                processed_tokens += batch_tokens
                active_count += len(feature_ids)
                progress.update(batch_tokens)

        if processed_tokens != token_count:
            raise RuntimeError(
                f"processed {processed_tokens:,} tokens; expected {token_count:,}"
            )

        feature_ids = (
            np.memmap(
                feature_temporary,
                mode="r",
                dtype=feature_dtype,
                shape=(active_count,),
            )
            if active_count
            else np.empty(0, dtype=feature_dtype)
        )
        active_values = (
            np.memmap(
                values_temporary,
                mode="r",
                dtype=ACTIVATION_VALUE_DTYPE,
                shape=(active_count,),
            )
            if active_count
            else np.empty(0, dtype=ACTIVATION_VALUE_DTYPE)
        )
        save_activations(
            output_path,
            metadata,
            evaluation_tokens,
            row_ptr,
            feature_ids,
            active_values,
            feature_counts,
            feature_max,
        )
    finally:
        if row_ptr is not None:
            row_ptr.flush()
            del row_ptr
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.sae_dir)
    cache = evaluation_cache(config, args.cache_dir)
    if cache is None:
        print("No local evaluation set is available; exiting.")
        return

    evaluation_path, available_tokens = cache
    token_count = (
        available_tokens
        if args.max_tokens is None
        else min(args.max_tokens, available_tokens)
    )
    evaluation_tokens = np.memmap(
        evaluation_path, mode="r", dtype=np.uint32, shape=(available_tokens,)
    )[:token_count]
    output_path = args.output or args.sae_dir / "activations"
    if output_path.exists():
        raise FileExistsError(f"activation output already exists: {output_path}")

    device = choose_device(args.device)
    tokenizer = load_tokenizer(config["model_id"])
    model = load_causal_lm(
        config["model_id"],
        getattr(torch, config["model_dtype"]),
        device,
    )
    _layer_path, layers = find_transformer_layers(model)
    layer_index = int(config["layer_index"])
    if not 0 <= layer_index < len(layers):
        raise ValueError(
            f"configured activation layer {layer_index} is not present in the model"
        )

    sae = load_sae(args.sae_dir, config, device)
    model_batch_size = args.model_batch_size or int(config["model_batch_size"])
    print(
        f"Device: {device} | Evaluation: {token_count:,} tokens | "
        f"Layer: {layer_index} | Output: {output_path}"
    )
    with ResidualStreamCapture(layers[layer_index]) as capture:
        write_activations(
            output_path,
            tokenizer.pad_token_id,
            model,
            capture,
            sae,
            evaluation_tokens,
            config,
            device,
            model_batch_size,
        )


if __name__ == "__main__":
    main()

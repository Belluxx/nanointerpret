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
    whole_contexts,
)
from src.experiment import sae_input
from src.runtime import (
    capture_layer_input,
    choose_device,
    find_transformer_layers,
    load_causal_lm,
)
from src.sae import FIRING_THRESHOLD, TopKSAE, load_sae


CACHE_DIR = Path("artifacts/token_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record SAE feature activations on the dedicated recording set.")
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.pt.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Token-cache directory populated during training. Default: artifacts/token_cache.")
    parser.add_argument("--tokens", type=int, default=None, help="Recording tokens to process, rounded down to whole contexts. Default: the full recording split.")
    parser.add_argument("--model-batch-size", type=int, default=None, help="Contexts processed together. Default: the training configuration.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <sae-dir>/activations.")
    return parser.parse_args()


def encode_activations(
    sae: TopKSAE,
    residuals: Tensor,
    activation_scale: float,
    passthrough_dims: list[int],
    batch_size: int,
    feature_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = sae_input(residuals, passthrough_dims)
    indices = []
    values = []
    for start in range(0, len(residuals), batch_size):
        x = residuals[start : start + batch_size].float()
        x.mul_(activation_scale)
        batch_indices, batch_values, _ = sae.encode(x)
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
    model,
    layer,
    sae: TopKSAE,
    recording_tokens: np.ndarray,
    config: dict,
    device: torch.device,
    model_batch_size: int,
) -> None:
    context_size = int(config["context_size"])
    token_count = len(recording_tokens)
    metadata = {
        "model_id": config["model_id"],
        "context_size": context_size,
        "layer_index": int(config["layer_index"]),
        "passthrough_dims": config["passthrough_dims"],
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
                recording_tokens, context_size, model_batch_size, shuffle=False, seed=0
            )
            for input_ids in batches:
                residuals = capture_layer_input(model, layer, input_ids.to(device)).flatten(0, 1)
                counts, feature_ids, active_values = encode_activations(
                    sae,
                    residuals,
                    float(config["activation_scale"]),
                    config["passthrough_dims"],
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
            recording_tokens,
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
    config = json.loads((args.sae_dir / "config.json").read_text())
    spec = TokenCacheSpec(
        cache_dir=args.cache_dir,
        model_id=config["model_id"],
        dataset_id=config["dataset_id"],
        dataset_config=config["dataset_config"],
        train_tokens=config["train_tokens"],
        validation_tokens=config["validation_tokens"],
        recording_tokens=config["recording_tokens"],
    )
    if spec.recording_tokens <= 0 or not token_cache_is_valid(spec, recording_only=True):
        raise FileNotFoundError(f"no recording split for this SAE in {args.cache_dir}")
    _, _, recording_path, _ = token_cache_paths(spec)
    available_tokens = spec.recording_tokens
    token_count = available_tokens if args.tokens is None else args.tokens
    if token_count > available_tokens:
        raise ValueError(
            f"--tokens requests {token_count:,} tokens, but the recording cache "
            f"contains {available_tokens:,}"
        )
    token_count = whole_contexts(token_count, int(config["context_size"]))
    recording_tokens = np.memmap(
        recording_path, mode="r", dtype=np.uint32, shape=(available_tokens,)
    )[:token_count]
    output_path = args.output or args.sae_dir / "activations"
    if output_path.exists():
        raise FileExistsError(f"activation output already exists: {output_path}")

    device = choose_device(args.device)
    model = load_causal_lm(
        config["model_id"],
        getattr(torch, config["model_dtype"]),
        device,
    )
    layers = find_transformer_layers(model)
    layer_index = int(config["layer_index"])
    sae = load_sae(args.sae_dir, config, device)
    model_batch_size = args.model_batch_size or int(config["model_batch_size"])
    print(
        f"Device: {device} | Recording: {token_count:,} tokens | "
        f"Layer: {layer_index} | Output: {output_path}"
    )
    write_activations(
        output_path,
        model,
        layers[layer_index],
        sae,
        recording_tokens,
        config,
        device,
        model_batch_size,
    )


if __name__ == "__main__":
    main()

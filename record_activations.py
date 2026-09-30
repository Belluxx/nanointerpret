"""Record SAE feature activations on the dedicated recording split, indexed by feature."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data import as_contexts, context_batches, token_cache
from src.experiment import load_sae
from src.runtime import capture_layer_input, choose_device, find_transformer_layers, load_causal_lm
from src.sae import FIRING_THRESHOLD, TopKSAE

VALUE_DTYPE = np.float16
INDEX_CHUNK = 1 << 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.pt.")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"), help="Token-cache directory populated during training. Default: artifacts/token_cache.")
    parser.add_argument("--tokens", type=int, default=None, help="Recording tokens to process, rounded down to whole contexts. Default: the full recording split.")
    parser.add_argument("--model-batch-size", type=int, default=None, help="Contexts processed together. Default: the training configuration.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <sae-dir>/activations.")
    return parser.parse_args()


def index_dtype(size: int) -> type[np.unsignedinteger]:
    return np.uint32 if size <= np.iinfo(np.uint32).max else np.uint64


def raw_dtypes(d_sae: int, token_count: int) -> tuple[np.dtype, type[np.unsignedinteger]]:
    return np.min_scalar_type(d_sae - 1), index_dtype(token_count)


@torch.inference_mode()
def record(
    model, layer, sae: TopKSAE, contexts: np.ndarray, batch_size: int, sae_batch_size: int, output: Path
) -> tuple[np.ndarray, np.ndarray]:
    # Append every firing activation, in token order, as raw (feature, position, value) files.
    feature_counts = np.zeros(sae.d_sae, dtype=np.int64)
    feature_max = np.zeros(sae.d_sae, dtype=np.float32)
    feature_dtype, position_dtype = raw_dtypes(sae.d_sae, contexts.size)
    position = 0
    with (
        (output / "features.raw").open("wb") as features_file,
        (output / "positions.raw").open("wb") as positions_file,
        (output / "values.raw").open("wb") as values_file,
        tqdm(total=contexts.size, unit="tok", desc="Record", dynamic_ncols=True) as progress,
    ):
        for ids in context_batches(len(contexts), batch_size):
            input_ids = torch.from_numpy(contexts[ids].astype(np.int64)).to(model.device)
            residual = capture_layer_input(model, layer, input_ids).flatten(0, 1)
            for x in sae.normalize(residual).split(sae_batch_size):
                indices, values, _ = sae.encode(x)
                firing = values > FIRING_THRESHOLD
                features = indices[firing].cpu().numpy().astype(feature_dtype)
                activations = values[firing].cpu().numpy().astype(VALUE_DTYPE)
                positions = firing.nonzero()[:, 0].cpu().numpy().astype(position_dtype) + position
                features.tofile(features_file)
                positions.tofile(positions_file)
                activations.tofile(values_file)
                feature_counts += np.bincount(features, minlength=sae.d_sae)
                np.maximum.at(feature_max, features, activations)
                position += len(x)
            progress.update(len(residual))
    return feature_counts, feature_max


def index_by_feature(output: Path, feature_counts: np.ndarray, token_count: int) -> None:
    # Counting sort of the raw activations into per-feature runs, still in token order within each run.
    feature_dtype, position_dtype = raw_dtypes(len(feature_counts), token_count)
    activation_count = int(feature_counts.sum())
    feature_ptr = np.zeros(len(feature_counts) + 1, dtype=index_dtype(activation_count))
    np.cumsum(feature_counts, out=feature_ptr[1:])
    np.save(output / "feature_ptr.npy", feature_ptr)

    features = np.memmap(output / "features.raw", dtype=feature_dtype, mode="r")
    positions = np.memmap(output / "positions.raw", dtype=position_dtype, mode="r")
    values = np.memmap(output / "values.raw", dtype=VALUE_DTYPE, mode="r")
    sorted_positions = np.lib.format.open_memmap(output / "token_positions.npy", "w+", position_dtype, (activation_count,))
    sorted_values = np.lib.format.open_memmap(output / "values.npy", "w+", VALUE_DTYPE, (activation_count,))
    cursors = feature_ptr[:-1].astype(np.int64)
    for start in tqdm(range(0, activation_count, INDEX_CHUNK), unit="chunk", desc="Index features", dynamic_ncols=True):
        chunk = slice(start, start + INDEX_CHUNK)
        order = np.argsort(features[chunk], kind="stable")
        chunk_features = features[chunk][order]
        chunk_counts = np.bincount(chunk_features, minlength=len(cursors))
        rank = np.arange(len(order)) - (np.cumsum(chunk_counts) - chunk_counts)[chunk_features]
        destination = cursors[chunk_features] + rank
        sorted_positions[destination] = positions[chunk][order]
        sorted_values[destination] = values[chunk][order]
        cursors += chunk_counts
    sorted_positions.flush()
    sorted_values.flush()
    del features, positions, values, sorted_positions, sorted_values
    for name in ("features", "positions", "values"):
        (output / f"{name}.raw").unlink()


def main() -> None:
    args = parse_args()
    output = args.output or args.sae_dir / "activations"
    if output.exists():
        raise FileExistsError(f"activation output already exists: {output}")
    device = choose_device(args.device)
    sae, config = load_sae(args.sae_dir, device)

    offset = config.train_tokens + config.validation_tokens
    tokens = token_cache(
        args.cache_dir, config.model_id, config.dataset_id, config.dataset_config, offset + config.recording_tokens
    )[offset:]
    if args.tokens is not None:
        if args.tokens > len(tokens):
            raise ValueError(f"--tokens requests {args.tokens:,} tokens, but the recording split has {len(tokens):,}")
        tokens = tokens[: args.tokens]
    contexts = as_contexts(tokens, config.context_size)
    if not contexts.size:
        raise ValueError("no recording tokens; train with --recording-tokens of at least one context")

    model = load_causal_lm(config.model_id, config.model_dtype, device)
    layer = find_transformer_layers(model)[config.layer_index]
    print(f"Device: {device} | Recording: {contexts.size:,} tokens | Layer: {config.layer_index} | Output: {output}")

    temporary = output.with_name(output.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        feature_counts, feature_max = record(
            model, layer, sae, contexts, args.model_batch_size or config.model_batch_size, config.sae_batch_size, temporary
        )
        index_by_feature(temporary, feature_counts, contexts.size)
        np.save(temporary / "token_ids.npy", contexts.reshape(-1))
        np.save(temporary / "feature_max.npy", feature_max)
        metadata = {
            "model_id": config.model_id,
            "context_size": config.context_size,
            "layer_index": config.layer_index,
            "passthrough_dims": config.passthrough_dims,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

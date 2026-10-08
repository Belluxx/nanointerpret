"""Record SAE feature activations on the dedicated recording split, indexed by token and by feature."""

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data import as_contexts, context_batches, token_cache
from src.experiment import load_sae
from src.runtime import capture_layer_input, choose_device, compile_layers_before, find_transformer_layers, load_causal_lm
from src.sae import FIRING_THRESHOLD, TopKSAE

VALUE_DTYPE = np.float16
INDEX_CHUNK = 1 << 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.pt.")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/token_cache"), help="Token-cache directory populated during training. Default: artifacts/token_cache.")
    parser.add_argument("--tokens", type=int, default=None, help="Recording tokens to process, rounded down to whole contexts. Default: the full recording split.")
    parser.add_argument("--model-batch-size", type=int, default=None, help="Contexts processed together. Default: the training configuration.")
    parser.add_argument("--no-compile-model", action="store_false", dest="compile_model", help="Disable compilation of transformer layers before the capture point on MPS.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <sae-dir>/activations.")
    return parser.parse_args()


def index_dtype(size: int) -> type[np.unsignedinteger]:
    return np.uint32 if size <= np.iinfo(np.uint32).max else np.uint64


def feature_dtype(d_sae: int) -> np.dtype:
    return np.min_scalar_type(d_sae - 1)


@torch.inference_mode()
def record(
    model, layer, sae: TopKSAE, contexts: np.ndarray, batch_size: int, sae_batch_size: int, output: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Append every firing activation, in token order, as raw (feature, value) files.
    feature_counts = np.zeros(sae.d_sae, dtype=np.int64)
    feature_max = np.zeros(sae.d_sae, dtype=np.float32)
    token_counts = np.zeros(contexts.size, dtype=np.int64)
    sink_positions = []
    position = 0
    with (
        (output / "features.raw").open("wb") as features_file,
        (output / "values.raw").open("wb") as values_file,
        tqdm(total=contexts.size, unit="tok", desc="Record", dynamic_ncols=True) as progress,
    ):
        for ids in context_batches(len(contexts), batch_size):
            input_ids = torch.from_numpy(contexts[ids].astype(np.int64)).to(model.device)
            residual = capture_layer_input(model, layer, input_ids).flatten(0, 1)
            sinks = sae.is_sink(residual)
            sink_positions.append(torch.nonzero(sinks).flatten().cpu().numpy() + position)
            for rows in torch.nonzero(~sinks).flatten().split(sae_batch_size):
                indices, values, _ = sae.encode(sae.normalize(residual[rows]))
                firing = values > FIRING_THRESHOLD
                features = indices[firing].cpu().numpy().astype(feature_dtype(sae.d_sae))
                activations = values[firing].cpu().numpy().astype(VALUE_DTYPE)
                features.tofile(features_file)
                activations.tofile(values_file)
                token_counts[rows.cpu().numpy() + position] = firing.sum(dim=1).cpu().numpy()
                feature_counts += np.bincount(features, minlength=sae.d_sae)
                np.maximum.at(feature_max, features, activations)
            position += len(residual)
            progress.update(len(residual))
    return feature_counts, feature_max, token_counts, np.concatenate(sink_positions)


def index(output: Path, feature_counts: np.ndarray, token_counts: np.ndarray) -> None:
    # Store the raw activations as token-major runs, and counting-sort them into feature-major runs
    # that stay in token order within each feature.
    activation_count = int(feature_counts.sum())
    feature_ptr = np.zeros(len(feature_counts) + 1, dtype=index_dtype(activation_count))
    np.cumsum(feature_counts, out=feature_ptr[1:])
    token_ptr = np.zeros(len(token_counts) + 1, dtype=feature_ptr.dtype)
    np.cumsum(token_counts, out=token_ptr[1:])
    np.save(output / "feature_ptr.npy", feature_ptr)
    np.save(output / "token_ptr.npy", token_ptr)

    def create(name: str, dtype: np.dtype) -> np.memmap:
        return np.lib.format.open_memmap(output / f"{name}.npy", "w+", dtype, (activation_count,))

    raw_features = np.memmap(output / "features.raw", dtype=feature_dtype(len(feature_counts)), mode="r")
    raw_values = np.memmap(output / "values.raw", dtype=VALUE_DTYPE, mode="r")
    token_features = create("token_features", raw_features.dtype)
    token_values = create("token_values", VALUE_DTYPE)
    feature_positions = create("feature_positions", index_dtype(len(token_counts)))
    feature_values = create("feature_values", VALUE_DTYPE)
    cursors = feature_ptr[:-1].astype(np.int64)
    for start in tqdm(range(0, activation_count, INDEX_CHUNK), unit="chunk", desc="Index", dynamic_ncols=True):
        chunk = slice(start, start + INDEX_CHUNK)
        features, values = np.array(raw_features[chunk]), np.array(raw_values[chunk])
        token_features[chunk], token_values[chunk] = features, values
        # Each activation belongs to the last token whose run starts at or before it.
        activation_ids = np.arange(start, start + len(features), dtype=token_ptr.dtype)
        positions = np.searchsorted(token_ptr, activation_ids, side="right") - 1
        order = np.argsort(features, kind="stable")
        sorted_features = features[order]
        chunk_counts = np.bincount(sorted_features, minlength=len(cursors))
        rank = np.arange(len(order)) - (np.cumsum(chunk_counts) - chunk_counts)[sorted_features]
        destination = cursors[sorted_features] + rank
        feature_positions[destination] = positions[order]
        feature_values[destination] = values[order]
        cursors += chunk_counts
    for array in (token_features, token_values, feature_positions, feature_values):
        array.flush()
    del raw_features, raw_values, token_features, token_values, feature_positions, feature_values
    for name in ("features", "values"):
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
    layers = find_transformer_layers(model)
    if args.compile_model and device.type == "mps":
        compile_layers_before(layers, config.layer_index)
    layer = layers[config.layer_index]
    print(f"Device: {device} | Recording: {contexts.size:,} tokens | Layer: {config.layer_index} | Output: {output}")

    temporary = output.with_name(output.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        feature_counts, feature_max, token_counts, sink_positions = record(
            model, layer, sae, contexts, args.model_batch_size or config.model_batch_size, config.sae_batch_size, temporary
        )
        index(temporary, feature_counts, token_counts)
        np.save(temporary / "token_ids.npy", contexts.reshape(-1))
        np.save(temporary / "sink_positions.npy", sink_positions.astype(index_dtype(contexts.size)))
        np.save(temporary / "feature_max.npy", feature_max)
        metadata = {
            "sae_dir": str(args.sae_dir.resolve()),
            "dataset_token_offset": offset,
            "token_count": contexts.size,
            "firing_threshold": FIRING_THRESHOLD,
            "sae_config": asdict(config),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

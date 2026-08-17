from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import groupby
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from visualize import ActivationData, STATIC_DIR


CONTEXTS_PER_FEATURE = 40
CONTEXT_TOKENS = 128
FEATURES_PER_FILE = 32
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact, static SAE feature visualizer."
    )
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--names",
        type=Path,
        help=(
            "Feature-title JSONL. Default: feature_names.jsonl next to the "
            "activation directory, when present."
        ),
    )
    parser.add_argument(
        "--intervention-url",
        help="Public intervention endpoint. The sandbox is hidden when omitted.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent export workers. Default: {DEFAULT_WORKERS}.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def render_context(
    data: ActivationData,
    contexts,
    group_index: int,
) -> dict:
    activation_start = int(contexts.group_starts[group_index])
    activation_stop = (
        int(contexts.group_starts[group_index + 1])
        if group_index + 1 < len(contexts.group_starts)
        else len(contexts.token_positions)
    )
    positions = contexts.token_positions[activation_start:activation_stop]
    values = contexts.activation_values[activation_start:activation_stop]
    focus = int(positions[np.argmax(values)])
    context_id = focus // data.context_size
    context_start = context_id * data.context_size
    context_stop = min(context_start + data.context_size, len(data.token_ids))
    window_start = max(context_start, focus - CONTEXT_TOKENS // 2)
    window_start = min(window_start, context_stop - CONTEXT_TOKENS)
    window_start = max(context_start, window_start)
    window_stop = min(window_start + CONTEXT_TOKENS, context_stop)
    included = (positions >= window_start) & (positions < window_stop)

    return {
        "context_id": context_id,
        "peak_activation": float(contexts.peaks[group_index]),
        "tokens": [
            data.decode_token(int(token_id))
            for token_id in data.token_ids[window_start:window_stop]
        ],
        "activation_positions": (
            positions[included] - window_start
        ).astype(int).tolist(),
        "activation_values": values[included].astype(float).tolist(),
    }


def representative_contexts(data: ActivationData, feature_id: int) -> list[dict]:
    contexts = data._feature_contexts(feature_id)
    ordered_peaks = contexts.peaks[contexts.peak_order]
    targets = np.linspace(0.0, float(ordered_peaks[-1]), CONTEXTS_PER_FEATURE)
    ranks = np.minimum(
        np.searchsorted(ordered_peaks, targets), len(ordered_peaks) - 1
    )
    group_indices = np.unique(contexts.peak_order[ranks])
    group_indices = group_indices[np.argsort(contexts.peaks[group_indices])[::-1]]
    return [render_context(data, contexts, int(index)) for index in group_indices]


def export_feature_file(
    data: ActivationData,
    output_directory: Path,
    shard: tuple[int, list[dict]],
) -> None:
    shard_id, features = shard
    payload = {}
    for feature in features:
        feature_id = feature["id"]
        feature_payload = data.feature(feature_id)
        feature_payload["contexts"] = representative_contexts(data, feature_id)
        payload[feature_id] = feature_payload
    write_json(output_directory / f"{shard_id}.json", payload)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"export output already exists: {args.output}")

    names_path = args.names
    if names_path is None:
        default_names = args.activations.with_name("feature_names.jsonl")
        names_path = default_names if default_names.exists() else None

    data = ActivationData(args.activations, names_path)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary export output already exists: {temporary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(STATIC_DIR, temporary)
        feature_directory = temporary / "data" / "features"
        feature_directory.mkdir(parents=True)
        config = {
            "dataDirectory": "data",
            "featuresPerFile": FEATURES_PER_FILE,
            "interventionUrl": args.intervention_url,
        }
        (temporary / "config.js").write_text(
            "window.NANOINTERPRET_CONFIG = "
            + json.dumps(config, ensure_ascii=False)
            + ";\n",
            encoding="utf-8",
        )
        write_json(temporary / "data" / "summary.json", data.summary())

        shards = [
            (shard_id, list(features))
            for shard_id, features in groupby(
                data.features,
                key=lambda feature: feature["id"] // FEATURES_PER_FILE,
            )
        ]
        export = partial(export_feature_file, data, feature_directory)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(export, shards)
            for _ in tqdm(
                results,
                total=len(shards),
                unit="file",
                desc="Export",
                dynamic_ncols=True,
            ):
                pass

        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Exported {len(data.features):,} features to {args.output}")


if __name__ == "__main__":
    main()

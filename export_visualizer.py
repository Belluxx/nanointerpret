from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from visualize import ActivationData, STATIC_DIR


CONTEXTS_PER_FEATURE = 20
CONTEXT_TOKENS = 64


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
    targets = np.linspace(
        0.0,
        float(ordered_peaks[-1]),
        CONTEXTS_PER_FEATURE,
    )
    ranks = np.minimum(
        np.searchsorted(ordered_peaks, targets),
        len(ordered_peaks) - 1,
    )
    group_indices = set(map(int, contexts.peak_order[ranks]))
    ordered = sorted(
        group_indices,
        key=lambda group_index: contexts.peaks[group_index],
        reverse=True,
    )
    return [render_context(data, contexts, group_index) for group_index in ordered]


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
            "interventionUrl": args.intervention_url,
        }
        (temporary / "config.js").write_text(
            "window.NANOINTERPRET_CONFIG = "
            + json.dumps(config, ensure_ascii=False)
            + ";\n",
            encoding="utf-8",
        )
        write_json(temporary / "data" / "summary.json", data.summary())

        for feature in tqdm(
            data.features,
            unit="feature",
            desc="Export",
            dynamic_ncols=True,
        ):
            feature_id = feature["id"]
            shard = feature_directory / f"{feature_id // 1000:03d}"
            shard.mkdir(exist_ok=True)
            payload = data.feature(feature_id)
            payload["contexts"] = representative_contexts(data, feature_id)
            write_json(shard / f"{feature_id}.json", payload)

        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Exported {len(data.features):,} features to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import groupby
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize import ActivationData, STATIC_DIR, existing_file


CONTEXTS_PER_FEATURE = 40
CONTEXT_TOKENS = 128
FEATURES_PER_FILE = 32
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact, static SAE feature visualizer.")
    parser.add_argument("--activations", type=Path, required=True, help="Activation directory produced by record_activations.py.")
    parser.add_argument("--names", type=Path, help="Feature-title JSONL. Default: feature_names.jsonl next to the activation directory, when present.")
    parser.add_argument("--feature-scores", type=Path, help="Feature-score JSONL produced by evaluate_features.py. Default: feature_scores.jsonl next to the activation directory, when present.")
    parser.add_argument("--starred-feature-threshold", type=float, default=0.9, help="Score at or above which a feature is marked high-quality and starred. Default: 0.9.")
    parser.add_argument("--intervention-url", help="Public intervention endpoint. The playground is hidden when omitted.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Concurrent export workers. Default: {DEFAULT_WORKERS}.")
    parser.add_argument("--output", type=Path, required=True, help="New directory in which to write the static visualizer bundle.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


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
        feature_payload["contexts"] = data.representative_contexts(
            feature_id,
            CONTEXTS_PER_FEATURE,
            CONTEXT_TOKENS,
        )
        payload[feature_id] = feature_payload
    write_json(output_directory / f"{shard_id}.json", payload)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"export output already exists: {args.output}")

    names_path = args.names or existing_file(
        args.activations.with_name("feature_names.jsonl")
    )
    scores_path = args.feature_scores or existing_file(
        args.activations.with_name("feature_scores.jsonl")
    )
    intervention_examples_path = existing_file(
        args.activations.parent / "intervention_examples.jsonl"
    )

    data = ActivationData(
        args.activations,
        names_path,
        scores_path=scores_path,
        intervention_examples_path=intervention_examples_path,
        starred_feature_threshold=args.starred_feature_threshold,
    )
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

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from transformers import AutoTokenizer

from src.data import load_analysis
from src.feature_examples import choose_activation_examples


STATIC_DIR = Path(__file__).with_name("visualizer")
FEATURE_ROUTE = re.compile(r"^/api/features/(\d+)$")
CONTEXT_LIMIT = 20
TOKENS_PER_PERCENTILE = 4
TOKEN_PERCENTILES = (95, 50, 25)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse SAE feature activations in a local web UI."
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--names",
        type=Path,
        help=(
            "Feature-title JSONL produced by interpret_features.py. "
            "Default: feature_names.jsonl next to the analysis directory, when present."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    return parser.parse_args()


def load_titles(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}

    titles = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                titles[int(result["feature_id"])] = str(result["title"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid feature title on line {line_number} of {path}"
                ) from error
    return titles


class AnalysisData:
    def __init__(self, analysis_path: Path, names_path: Path | None, tokenizer=None):
        analysis = load_analysis(analysis_path)
        self.metadata = analysis.metadata
        self.token_ids = analysis.token_ids
        self.feature_ptr = analysis.feature_ptr
        self.token_positions = analysis.token_positions
        self.values = analysis.values
        self.feature_max = analysis.feature_max

        self.d_sae = len(self.feature_ptr) - 1
        self.context_size = int(self.metadata["context_size"])
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.metadata["model_id"]
        )

        titles = load_titles(names_path)
        counts = np.diff(self.feature_ptr)
        self.features = [
            {
                "id": int(feature_id),
                "title": titles.get(int(feature_id)),
                "activation_count": int(counts[feature_id]),
                "max_activation": float(self.feature_max[feature_id]),
            }
            for feature_id in np.flatnonzero(counts)
        ]

    def summary(self) -> dict:
        return {
            "metadata": {
                "model_id": self.metadata["model_id"],
                "processed_tokens": len(self.token_ids),
                "layer_index": self.metadata["layer_index"],
                "d_sae": self.d_sae,
            },
            "features": self.features,
        }

    @lru_cache(maxsize=None)
    def decode_token(self, token_id: int) -> str:
        text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if text:
            return text
        return str(self.tokenizer.convert_ids_to_tokens(token_id))

    def feature(self, feature_id: int) -> dict:
        if not 0 <= feature_id < self.d_sae:
            raise KeyError(feature_id)

        start, stop = map(int, self.feature_ptr[feature_id : feature_id + 2])
        if start == stop:
            raise KeyError(feature_id)

        token_positions = self.token_positions[start:stop]
        activation_values = self.values[start:stop]
        activating_token_ids = self.token_ids[token_positions]
        unique_token_ids, token_groups, token_counts = np.unique(
            activating_token_ids, return_inverse=True, return_counts=True
        )
        token_maxima = np.zeros(len(unique_token_ids), dtype=np.float32)
        np.maximum.at(token_maxima, token_groups, activation_values)
        token_sums = np.bincount(token_groups, weights=activation_values)
        token_means = token_sums / token_counts

        token_summaries = []
        for percentile in TOKEN_PERCENTILES:
            percentile_activation = float(
                np.percentile(activation_values, percentile)
            )
            selected_token_groups = []
            for activation_index in np.argsort(
                np.abs(activation_values - percentile_activation), kind="stable"
            ):
                token_group = int(token_groups[activation_index])
                if token_group not in selected_token_groups:
                    selected_token_groups.append(token_group)
                if len(selected_token_groups) == TOKENS_PER_PERCENTILE:
                    break

            token_summaries.append(
                {
                    "percentile": percentile,
                    "tokens": [
                        {
                            "token_id": int(unique_token_ids[index]),
                            "token": self.decode_token(
                                int(unique_token_ids[index])
                            ),
                            "activation_count": int(token_counts[index]),
                            "mean_activation": float(token_means[index]),
                            "max_activation": float(token_maxima[index]),
                        }
                        for index in selected_token_groups
                    ],
                }
            )

        context_ids = token_positions // self.context_size

        group_starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(context_ids)) + 1)
        )
        grouped_context_ids = context_ids[group_starts]
        peaks = np.maximum.reduceat(activation_values, group_starts)
        occurrences = np.diff(np.append(group_starts, len(context_ids)))

        def render_context(group_index: int, sample=None) -> dict:
            context_id = int(grouped_context_ids[group_index])
            context_start = context_id * self.context_size
            context_stop = min(
                context_start + self.context_size, len(self.token_ids)
            )
            token_slice = self.token_ids[context_start:context_stop]

            activation_start = int(group_starts[group_index])
            activation_stop = activation_start + int(occurrences[group_index])
            positions = (
                token_positions[activation_start:activation_stop] - context_start
            )
            context_activations = np.zeros(len(token_slice), dtype=np.float32)
            context_activations[positions] = activation_values[
                activation_start:activation_stop
            ]

            context = {
                "context_id": context_id,
                "peak_activation": float(peaks[group_index]),
                "activation_count": int(occurrences[group_index]),
                "tokens": [
                    self.decode_token(int(token_id)) for token_id in token_slice
                ],
                "activations": context_activations.tolist(),
            }
            if sample is not None:
                context["sample"] = {
                    "bucket": sample.bucket,
                    "activation": sample.activation,
                    "percentile": sample.percentile,
                    "target_position": sample.token_position - context_start,
                }
            return context

        strongest = [
            render_context(int(group_index))
            for group_index in np.argsort(-peaks, kind="stable")[:CONTEXT_LIMIT]
        ]
        stratified = []
        samples = choose_activation_examples(
            feature_id,
            token_positions,
            activation_values,
            self.context_size,
        )
        for sample in samples or []:
            group_index = int(
                np.searchsorted(
                    grouped_context_ids,
                    sample.token_position // self.context_size,
                )
            )
            stratified.append(render_context(group_index, sample))

        return {
            "activation_count": int(len(token_positions)),
            "context_count": int(len(grouped_context_ids)),
            "token_groups": token_summaries,
            "contexts": {
                "strongest": strongest,
                "stratified": stratified,
            },
        }


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, data: AnalysisData, **kwargs):
        self.data = data
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/api/summary":
            self.send_json(self.data.summary())
            return

        match = FEATURE_ROUTE.fullmatch(request.path)
        if match:
            try:
                payload = self.data.feature(int(match.group(1)))
            except KeyError:
                self.send_json({"error": "feature not found"}, status=404)
            else:
                self.send_json(payload)
            return

        super().do_GET()

    def send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    args = parse_args()
    names_path = args.names
    if names_path is None:
        default_names = args.analysis.with_name("feature_names.jsonl")
        names_path = default_names if default_names.exists() else None

    print(f"Loading {args.analysis} ...")
    data = AnalysisData(args.analysis, names_path)
    handler = partial(VisualizerHandler, data=data)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{server.server_address[1]}"
        print(
            f"Loaded {len(data.features):,} active features across "
            f"{len(data.token_ids):,} tokens."
        )
        print(f"Open {url} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping visualizer.")


if __name__ == "__main__":
    main()

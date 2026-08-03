from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from transformers import AutoTokenizer

from src.feature_examples import choose_activation_examples


ANALYSIS_PATH = Path("artifacts/sae_gemma_3_270m/analysis.npz")
STATIC_DIR = Path(__file__).with_name("visualizer")
CONTEXT_ROUTE = re.compile(r"^/api/features/(\d+)/contexts$")
PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
TOKENS_PER_PERCENTILE = 4
TOKEN_PERCENTILES = (("high", 95), ("med", 50), ("low", 25))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse SAE feature activations in a local web UI."
    )
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument(
        "--names",
        type=Path,
        help=(
            "Feature-title JSONL produced by interpret_features.py. "
            "Default: feature_names.jsonl next to the analysis file, when present."
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
        with np.load(analysis_path) as analysis:
            self.metadata = json.loads(analysis["metadata"].item())
            self.token_ids = analysis["token_ids"]
            self.context_ptr = analysis["context_ptr"]
            self.row_ptr = analysis["row_ptr"]
            self.feature_ids = analysis["feature_ids"]
            self.values = analysis["values"]

        if self.metadata.get("format") != "csr":
            raise ValueError("the visualizer requires a CSR analysis artifact")

        self.d_sae = int(self.metadata["d_sae"])
        self.titles = load_titles(names_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.metadata["model_id"]
        )

        counts = np.bincount(self.feature_ids, minlength=self.d_sae)
        maxima = np.zeros(self.d_sae, dtype=np.float32)
        np.maximum.at(maxima, self.feature_ids, self.values)
        self.features = [
            {
                "id": int(feature_id),
                "title": self.titles.get(int(feature_id)),
                "activation_count": int(counts[feature_id]),
                "max_activation": float(maxima[feature_id]),
            }
            for feature_id in np.flatnonzero(counts)
        ]

    def summary(self) -> dict:
        metadata_keys = (
            "model_id",
            "dataset_id",
            "processed_tokens",
            "context_count",
            "context_size",
            "layer_index",
            "residual_location",
            "d_sae",
            "k",
            "created_at",
        )
        return {
            "metadata": {
                key: self.metadata[key]
                for key in metadata_keys
                if key in self.metadata
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

    def feature_contexts(
        self,
        feature_id: int,
        offset: int = 0,
        limit: int = PAGE_SIZE,
        view: str = "strongest",
    ) -> dict:
        if not 0 <= feature_id < self.d_sae:
            raise KeyError(feature_id)
        if offset < 0 or limit <= 0:
            raise ValueError("offset must be nonnegative and limit must be positive")
        if view not in ("strongest", "stratified"):
            raise ValueError("view must be 'strongest' or 'stratified'")
        limit = min(limit, MAX_PAGE_SIZE)

        activation_indices = np.flatnonzero(self.feature_ids == feature_id)
        if not len(activation_indices):
            raise KeyError(feature_id)

        token_positions = np.searchsorted(
            self.row_ptr, activation_indices, side="right"
        ) - 1
        activation_values = self.values[activation_indices]
        activating_token_ids = self.token_ids[token_positions]
        unique_token_ids, token_groups, token_counts = np.unique(
            activating_token_ids, return_inverse=True, return_counts=True
        )
        token_maxima = np.zeros(len(unique_token_ids), dtype=np.float32)
        np.maximum.at(token_maxima, token_groups, activation_values)
        token_sums = np.bincount(token_groups, weights=activation_values)
        token_means = token_sums / token_counts

        activation_token_groups = []
        used_token_groups = set()
        for level, percentile in TOKEN_PERCENTILES:
            percentile_activation = float(
                np.percentile(activation_values, percentile)
            )
            nearby_activations = np.argsort(
                np.abs(activation_values - percentile_activation), kind="stable"
            )
            selected_token_groups = []
            for allow_reuse in (False, True):
                for activation_index in nearby_activations:
                    token_group = int(token_groups[activation_index])
                    if token_group in selected_token_groups:
                        continue
                    if not allow_reuse and token_group in used_token_groups:
                        continue
                    selected_token_groups.append(token_group)
                    if len(selected_token_groups) == TOKENS_PER_PERCENTILE:
                        break
                if len(selected_token_groups) == TOKENS_PER_PERCENTILE:
                    break
            used_token_groups.update(selected_token_groups)

            activation_token_groups.append(
                {
                    "level": level,
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

        context_ids = np.searchsorted(
            self.context_ptr, token_positions, side="right"
        ) - 1

        group_starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(context_ids)) + 1)
        )
        grouped_context_ids = context_ids[group_starts]
        peaks = np.maximum.reduceat(activation_values, group_starts)
        occurrences = np.diff(np.append(group_starts, len(context_ids)))

        def render_context(group_index: int, sample=None) -> dict:
            context_id = int(grouped_context_ids[group_index])
            context_start = int(self.context_ptr[context_id])
            context_stop = int(self.context_ptr[context_id + 1])
            token_slice = self.token_ids[context_start:context_stop]

            activation_start = int(
                np.searchsorted(token_positions, context_start, side="left")
            )
            activation_stop = int(
                np.searchsorted(token_positions, context_stop, side="left")
            )
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
                local_index = int(
                    np.searchsorted(activation_indices, sample.activation_index)
                )
                category = "stratified"
                if sample.bucket == "Top":
                    category = "top"
                elif sample.bucket == "Random positive":
                    category = "random"
                context["sample"] = {
                    "category": category,
                    "bucket": sample.bucket,
                    "activation": sample.activation,
                    "percentile": sample.percentile,
                    "target_position": int(
                        token_positions[local_index] - context_start
                    ),
                }
            return context

        stratified_available = True
        if view == "strongest":
            ranked_groups = np.argsort(-peaks, kind="stable")
            selected_groups = ranked_groups[offset : offset + limit]
            contexts = [
                render_context(int(group_index)) for group_index in selected_groups
            ]
        else:
            samples = choose_activation_examples(
                feature_id,
                activation_indices,
                self.values,
                self.context_ptr,
                self.row_ptr,
            )
            stratified_available = samples is not None
            contexts = []
            for sample in samples or []:
                local_index = int(
                    np.searchsorted(activation_indices, sample.activation_index)
                )
                group_index = int(
                    np.searchsorted(grouped_context_ids, context_ids[local_index])
                )
                contexts.append(render_context(group_index, sample))

        return {
            "feature_id": feature_id,
            "view": view,
            "activation_count": int(len(activation_indices)),
            "mean_activation": float(np.mean(activation_values)),
            "context_count": int(len(grouped_context_ids)),
            "unique_token_count": int(len(unique_token_ids)),
            "activation_token_groups": activation_token_groups,
            "stratified_available": stratified_available,
            "offset": offset,
            "contexts": contexts,
        }


class VisualizerHandler(BaseHTTPRequestHandler):
    data: AnalysisData

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/api/summary":
            self.send_json(self.data.summary())
            return

        match = CONTEXT_ROUTE.fullmatch(request.path)
        if match:
            query = parse_qs(request.query)
            try:
                offset = int(query.get("offset", [0])[0])
                limit = int(query.get("limit", [PAGE_SIZE])[0])
                view = query.get("view", ["strongest"])[0]
                payload = self.data.feature_contexts(
                    int(match.group(1)), offset=offset, limit=limit, view=view
                )
            except ValueError as error:
                self.send_json({"error": str(error)}, status=400)
            except KeyError:
                self.send_json({"error": "feature not found"}, status=404)
            else:
                self.send_json(payload)
            return

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        static_file = static_files.get(request.path)
        if static_file is None:
            self.send_error(404)
            return

        filename, content_type = static_file
        content = (STATIC_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

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
    handler = type("Handler", (VisualizerHandler,), {"data": data})
    server = ThreadingHTTPServer((args.host, args.port), handler)
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
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

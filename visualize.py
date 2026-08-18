from __future__ import annotations

import argparse
import json
import math
import re
import threading
from functools import cache, lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit

import numpy as np
from transformers import AutoTokenizer

from src.data import load_activations
from src.interventions import InterventionGenerator, InterventionRequest
from src.runtime import choose_device

STATIC_DIR = Path(__file__).with_name("visualizer")
FEATURE_ROUTE = re.compile(r"^/api/features/(\d+)$")
INTERVENTION_ROUTE = "/api/interventions/generate"
CONTEXT_LIMIT = 20
ACTIVATION_HISTOGRAM_BINS = 40
TOKENS_PER_PERCENTILE = 4
TOKEN_PERCENTILES = (95, 50, 25)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse SAE feature activations in a local web UI.")
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--names", type=Path, help="Feature-title JSONL produced by interpret_features.py. Default: feature_names.jsonl next to the activations directory, when present.")
    parser.add_argument("--sae-dir", type=Path, help="Training output containing config.json and sae_final.pt. Default: the activations directory's parent.")
    parser.add_argument("--feature-scores", type=Path, help="Feature-score JSONL produced by evaluate_features.py. Default: feature_scores.jsonl in --sae-dir, when present.")
    parser.add_argument("--starred-feature-threshold", type=unit_float, default=0.6, help="Score at or above which a feature is marked high-quality and starred. Default: 0.6.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    return parser.parse_args()


def load_titles(path: Path | None) -> dict[int, str | None]:
    if path is None:
        return {}

    titles = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                title = result["title"]
                if title is not None and not isinstance(title, str):
                    raise TypeError("feature title must be a string or null")
                titles[int(result["feature_id"])] = title
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid feature title on line {line_number} of {path}"
                ) from error
    return titles


def load_feature_scores(path: Path | None) -> dict[int, float | None]:
    if path is None:
        return {}

    scores = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                score = result["score"]
                if score is not None:
                    score = float(score)
                    if not math.isfinite(score):
                        raise ValueError
                scores[int(result["feature_id"])] = score
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid feature score on line {line_number} of {path}"
                ) from error
    return scores


class FeatureContexts(NamedTuple):
    token_positions: np.ndarray
    activation_values: np.ndarray
    group_starts: np.ndarray
    peaks: np.ndarray
    peak_order: np.ndarray


class ActivationData:
    def __init__(
        self,
        activations_path: Path,
        names_path: Path | None,
        scores_path: Path | None = None,
        starred_feature_threshold: float = 0.6,
        tokenizer=None,
    ):
        activations = load_activations(activations_path)
        self.metadata = activations.metadata
        self.token_ids = activations.token_ids
        self.feature_ptr = activations.feature_ptr
        self.token_positions = activations.token_positions
        self.values = activations.values

        self.d_sae = len(self.feature_ptr) - 1
        self.context_size = int(self.metadata["context_size"])
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.metadata["model_id"]
        )

        titles = load_titles(names_path)
        self.scores = load_feature_scores(scores_path)
        self.starred_feature_threshold = starred_feature_threshold
        counts = np.diff(self.feature_ptr)
        self.features = []
        for feature_id in np.flatnonzero(counts):
            feature_id = int(feature_id)
            score = self.scores.get(feature_id)
            self.features.append(
                {
                    "id": feature_id,
                    "title": titles.get(feature_id),
                    "score": score,
                    "high_quality": score is not None and score >= starred_feature_threshold,
                    "activation_count": int(counts[feature_id]),
                    "max_activation": float(activations.feature_max[feature_id]),
                }
            )

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

    @cache
    def decode_token(self, token_id: int) -> str:
        text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if text:
            return text
        return str(self.tokenizer.convert_ids_to_tokens(token_id))

    @lru_cache(maxsize=32)
    def _feature_contexts(self, feature_id: int) -> FeatureContexts:
        if not 0 <= feature_id < self.d_sae:
            raise KeyError(feature_id)

        start, stop = map(int, self.feature_ptr[feature_id : feature_id + 2])
        if start == stop:
            raise KeyError(feature_id)

        token_positions = self.token_positions[start:stop]
        activation_values = self.values[start:stop]
        context_ids = token_positions // self.context_size
        group_starts = np.concatenate(([0], np.flatnonzero(np.diff(context_ids)) + 1))
        peaks = np.maximum.reduceat(activation_values, group_starts)
        return FeatureContexts(
            token_positions,
            activation_values,
            group_starts,
            peaks,
            np.argsort(peaks, kind="stable"),
        )

    def _render_context(self, contexts: FeatureContexts, group_index: int) -> dict:
        activation_start = int(contexts.group_starts[group_index])
        activation_stop = (
            int(contexts.group_starts[group_index + 1])
            if group_index + 1 < len(contexts.group_starts)
            else len(contexts.token_positions)
        )
        context_id = int(
            contexts.token_positions[activation_start] // self.context_size
        )
        context_start = context_id * self.context_size
        context_stop = min(context_start + self.context_size, len(self.token_ids))
        token_slice = self.token_ids[context_start:context_stop]

        positions = (
            contexts.token_positions[activation_start:activation_stop] - context_start
        )
        context_activations = np.zeros(len(token_slice), dtype=np.float32)
        context_activations[positions] = contexts.activation_values[
            activation_start:activation_stop
        ]

        return {
            "context_id": context_id,
            "peak_activation": float(contexts.peaks[group_index]),
            "tokens": [self.decode_token(int(token_id)) for token_id in token_slice],
            "activations": context_activations.tolist(),
        }

    def feature(self, feature_id: int) -> dict:
        contexts = self._feature_contexts(feature_id)
        activating_token_ids = self.token_ids[contexts.token_positions]
        score = self.scores.get(feature_id)

        token_summaries = []
        for percentile in TOKEN_PERCENTILES:
            target = np.percentile(contexts.activation_values, percentile)
            selected_token_ids = []
            for activation_index in np.argsort(
                np.abs(contexts.activation_values - target), kind="stable"
            ):
                token_id = int(activating_token_ids[activation_index])
                if token_id not in selected_token_ids:
                    selected_token_ids.append(token_id)
                if len(selected_token_ids) == TOKENS_PER_PERCENTILE:
                    break

            token_summaries.append(
                {
                    "percentile": percentile,
                    "tokens": [
                        self.decode_token(token_id) for token_id in selected_token_ids
                    ],
                }
            )

        maximum = float(contexts.peaks.max())
        bin_count = min(ACTIVATION_HISTOGRAM_BINS, len(contexts.peaks))
        histogram, _ = np.histogram(
            contexts.peaks, bins=bin_count, range=(0.0, maximum)
        )

        return {
            "score": score,
            "high_quality": (
                score is not None and score >= self.starred_feature_threshold
            ),
            "activation_count": len(contexts.token_positions),
            "context_count": len(contexts.group_starts),
            "token_groups": token_summaries,
            "activation_histogram": histogram.tolist(),
        }

    def range_contexts(self, feature_id: int, minimum: float, maximum: float) -> dict:
        contexts = self._feature_contexts(feature_id)
        ordered_peaks = contexts.peaks[contexts.peak_order]
        minimum = ordered_peaks.dtype.type(minimum)
        maximum = ordered_peaks.dtype.type(maximum)
        start = int(np.searchsorted(ordered_peaks, minimum, side="left"))
        stop = int(np.searchsorted(ordered_peaks, maximum, side="right"))
        ordered = contexts.peak_order[start:stop]
        matching_count = len(ordered)
        if matching_count > CONTEXT_LIMIT:
            sample_indices = np.linspace(
                0, matching_count - 1, CONTEXT_LIMIT, dtype=int
            )
            ordered = ordered[sample_indices]

        ordered = ordered[::-1]

        return {
            "matching_context_count": matching_count,
            "contexts": [
                self._render_context(contexts, int(group_index))
                for group_index in ordered
            ],
        }


class InterventionPlayground:
    def __init__(
        self,
        sae_dir: Path,
        data: ActivationData,
        device_name: str,
    ):
        self.sae_dir = sae_dir
        self.data = data
        self.device = choose_device(device_name)
        self.generator: InterventionGenerator | None = None
        self.lock = threading.Lock()

    def generate(self, payload: dict) -> dict:
        request = InterventionRequest(**payload)
        with self.lock:
            if self.generator is None:
                print(f"Loading intervention model on {self.device} ...")
                self.generator = InterventionGenerator.from_sae_dir(
                    self.sae_dir, self.device, tokenizer=self.data.tokenizer
                )
            return self.generator.generate_pair(request)


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        data: ActivationData,
        playground: InterventionPlayground,
        **kwargs,
    ):
        self.data = data
        self.playground = playground
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/api/summary":
            self.send_json(self.data.summary())
            return

        match = FEATURE_ROUTE.fullmatch(request.path)
        if match:
            try:
                feature_id = int(match.group(1))
                if request.query:
                    query = parse_qs(request.query, strict_parsing=True)
                    minimum = float(query.get("min", [None])[0])
                    maximum = float(query.get("max", [None])[0])
                    if not all(map(math.isfinite, (minimum, maximum))):
                        raise ValueError
                    if minimum > maximum:
                        raise ValueError
                    payload = self.data.range_contexts(feature_id, minimum, maximum)
                else:
                    payload = self.data.feature(feature_id)
            except (TypeError, ValueError):
                self.send_json({"error": "invalid activation range"}, status=400)
            except KeyError:
                self.send_json({"error": "feature not found"}, status=404)
            else:
                self.send_json(payload)
            return

        super().do_GET()

    def do_POST(self) -> None:
        request = urlsplit(self.path)
        if request.path != INTERVENTION_ROUTE:
            self.send_json({"error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            result = self.playground.generate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "request body must be valid JSON"}, status=400)
        except Exception as error:
            self.log_error("intervention generation failed: %s", error)
            self.send_json({"error": str(error)}, status=500)
        else:
            self.send_json(result)

    def send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    args = parse_args()
    sae_dir = args.sae_dir or args.activations.parent
    names_path = args.names
    if names_path is None:
        default_names = args.activations.with_name("feature_names.jsonl")
        names_path = default_names if default_names.exists() else None
    scores_path = args.feature_scores
    if scores_path is None:
        default_scores = sae_dir / "feature_scores.jsonl"
        scores_path = default_scores if default_scores.exists() else None

    print(f"Loading {args.activations} ...")
    data = ActivationData(
        args.activations,
        names_path,
        scores_path=scores_path,
        starred_feature_threshold=args.starred_feature_threshold,
    )
    playground = InterventionPlayground(sae_dir, data, args.device)
    handler = partial(VisualizerHandler, data=data, playground=playground)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{server.server_address[1]}"
        print(f"Open {url}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping visualizer.")


if __name__ == "__main__":
    main()

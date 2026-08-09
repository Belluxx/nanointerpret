from __future__ import annotations

import argparse
import json
import math
import re
import threading
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from transformers import AutoTokenizer

from src.data import load_analysis
from src.interventions import (
    MAX_NEW_TOKENS,
    InterventionGenerator,
    InterventionRequest,
)
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
    parser.add_argument(
        "--sae-dir",
        type=Path,
        help=(
            "Training output containing config.json and sae_final.pt. "
            "Default: the analysis directory's parent."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
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

    @lru_cache(maxsize=32)
    def _context_data(self, feature_id: int) -> tuple:
        if not 0 <= feature_id < self.d_sae:
            raise KeyError(feature_id)

        start, stop = map(int, self.feature_ptr[feature_id : feature_id + 2])
        if start == stop:
            raise KeyError(feature_id)

        token_positions = self.token_positions[start:stop]
        activation_values = self.values[start:stop]
        context_ids = token_positions // self.context_size
        group_starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(context_ids)) + 1)
        )
        peaks = np.maximum.reduceat(activation_values, group_starts)
        return (
            token_positions,
            activation_values,
            group_starts,
            peaks,
            np.argsort(peaks, kind="stable"),
        )

    def _render_context(self, context_data: tuple, group_index: int) -> dict:
        (
            token_positions,
            activation_values,
            group_starts,
            peaks,
            _,
        ) = context_data
        activation_start = int(group_starts[group_index])
        activation_stop = (
            int(group_starts[group_index + 1])
            if group_index + 1 < len(group_starts)
            else len(token_positions)
        )
        context_id = int(token_positions[activation_start] // self.context_size)
        context_start = context_id * self.context_size
        context_stop = min(context_start + self.context_size, len(self.token_ids))
        token_slice = self.token_ids[context_start:context_stop]

        positions = token_positions[activation_start:activation_stop] - context_start
        context_activations = np.zeros(len(token_slice), dtype=np.float32)
        context_activations[positions] = activation_values[
            activation_start:activation_stop
        ]

        return {
            "context_id": context_id,
            "peak_activation": float(peaks[group_index]),
            "activation_count": activation_stop - activation_start,
            "tokens": [self.decode_token(int(token_id)) for token_id in token_slice],
            "activations": context_activations.tolist(),
        }

    def feature(self, feature_id: int) -> dict:
        context_data = self._context_data(feature_id)
        token_positions, activation_values, group_starts, peaks, _ = context_data
        activating_token_ids = self.token_ids[token_positions]

        token_summaries = []
        for percentile in TOKEN_PERCENTILES:
            target = np.percentile(activation_values, percentile)
            selected_token_ids = []
            for activation_index in np.argsort(
                np.abs(activation_values - target), kind="stable"
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
                        self.decode_token(token_id)
                        for token_id in selected_token_ids
                    ],
                }
            )

        strongest = [
            self._render_context(context_data, int(group_index))
            for group_index in np.argsort(-peaks, kind="stable")[:CONTEXT_LIMIT]
        ]

        maximum = float(peaks.max())
        bin_count = min(ACTIVATION_HISTOGRAM_BINS, len(peaks))
        histogram, _ = np.histogram(peaks, bins=bin_count, range=(0.0, maximum))

        return {
            "activation_count": int(len(token_positions)),
            "context_count": int(len(group_starts)),
            "token_groups": token_summaries,
            "activation_histogram": histogram.tolist(),
            "strongest_contexts": strongest,
        }

    def range_contexts(
        self, feature_id: int, minimum: float, maximum: float
    ) -> dict:
        context_data = self._context_data(feature_id)
        _, _, _, peaks, peak_order = context_data
        ordered_peaks = peaks[peak_order]
        minimum = ordered_peaks.dtype.type(minimum)
        maximum = ordered_peaks.dtype.type(maximum)
        start = int(np.searchsorted(ordered_peaks, minimum, side="left"))
        stop = int(np.searchsorted(ordered_peaks, maximum, side="right"))
        ordered = peak_order[start:stop]
        matching_count = len(ordered)
        if matching_count > CONTEXT_LIMIT:
            matching_peaks = peaks[ordered]
            targets = np.linspace(matching_peaks[0], matching_peaks[-1], CONTEXT_LIMIT)
            right = np.searchsorted(matching_peaks, targets).clip(0, matching_count - 1)
            left = np.maximum(right - 1, 0)
            nearest = np.where(
                targets - matching_peaks[left]
                <= matching_peaks[right] - targets,
                left,
                right,
            )
            offsets = np.arange(CONTEXT_LIMIT)
            nearest = np.minimum(nearest, matching_count - CONTEXT_LIMIT + offsets)
            selected = np.maximum.accumulate(nearest - offsets) + offsets
            ordered = ordered[selected]

        return {
            "matching_context_count": int(matching_count),
            "contexts": [
                self._render_context(context_data, int(group_index))
                for group_index in ordered
            ],
        }


class InterventionSandbox:
    def __init__(
        self,
        sae_dir: Path,
        data: AnalysisData,
        device_name: str,
    ):
        self.sae_dir = sae_dir
        self.data = data
        self.device = choose_device(device_name)
        self.generator: InterventionGenerator | None = None
        self.lock = threading.Lock()

    def generate(self, payload: object) -> dict:
        request = InterventionRequest.from_payload(payload, self.data.d_sae)
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
        data: AnalysisData,
        sandbox: InterventionSandbox,
        **kwargs,
    ):
        self.data = data
        self.sandbox = sandbox
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/api/summary":
            payload = self.data.summary()
            payload["max_new_tokens"] = MAX_NEW_TOKENS
            self.send_json(payload)
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
                    payload = self.data.range_contexts(
                        feature_id, minimum, maximum
                    )
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
            result = self.sandbox.generate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "request body must be valid JSON"}, status=400)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:
            self.log_error("intervention generation failed: %s", error)
            self.send_json({"error": str(error)}, status=500)
        else:
            self.send_json(result)

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
    sae_dir = args.sae_dir or args.analysis.parent
    sandbox = InterventionSandbox(sae_dir, data, args.device)
    handler = partial(VisualizerHandler, data=data, sandbox=sandbox)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{server.server_address[1]}"
        print(f"Open {url}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping visualizer.")


if __name__ == "__main__":
    main()

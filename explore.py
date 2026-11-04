"""Serve a local web explorer over the recorded SAE feature activations, with live steering."""

import argparse
import json
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.features import FeatureIndex
from src.interpret import CONTINUATION_LADDER, Probe, load_probe

STATIC_DIR = Path(__file__).parent / "explorer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.safetensors.")
    parser.add_argument("--activations", type=Path, default=None, help="Recorded activation directory. Default: <sae-dir>/activations.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto", help="Device for steering and the one-time feature summary.")
    return parser.parse_args()


class LazyProbe:
    """The model and SAE, loaded by the first request that needs them: browsing the index takes neither."""

    def __init__(self, index: FeatureIndex, sae_dir: Path, device: str):
        self.index, self.sae_dir, self.device = index, sae_dir, device
        self.lock = threading.Lock()
        self.probe = None

    def __call__(self) -> Probe:
        with self.lock:
            if self.probe is None:
                self.probe = load_probe(self.index, self.sae_dir, self.device)
            return self.probe


class ExplorerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, index: FeatureIndex, probe: LazyProbe, **kwargs):
        self.index = index
        self.probe = probe
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
            if url.path == "/api/steer":
                self.steer(int(query["feature"]), query["prompt"])
            else:
                self.send_json(self.route(url.path, query))
        except (KeyError, ValueError, IndexError, FileNotFoundError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def route(self, path: str, query: dict[str, str]) -> object:
        if path == "/api/overview":
            return self.index.overview()
        if path == "/api/features":
            low, high = (float(query["low"]), float(query["high"])) if "low" in query else (None, None)
            return self.index.browse(query["sort"], int(query["offset"]), int(query["limit"]), low, high, "trusted" in query, query.get("q", ""))
        if path.startswith("/api/feature/"):
            return self.index.feature(int(path.removeprefix("/api/feature/")))
        if path == "/api/token":
            return self.index.token_search(query["q"])
        raise KeyError(f"unknown endpoint {path}")

    def steer(self, feature: int, prompt: str) -> None:
        # Streams one JSON line per event: the strengths, then each step's tokens. Features without an
        # interpretation are calibrated here; a client that hangs up ends the generation at its next token.
        probe = self.probe()
        record = self.index.interpretations.get(feature)
        calibrated = record["steering"]["strength"] if record else probe.calibrated_strength(feature)
        strengths = [calibrated * multiple for multiple in CONTINUATION_LADDER]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            self.send_line({"strengths": strengths})
            for rows in probe.continuations(feature, prompt, strengths):
                self.send_line(rows)
        except ConnectionError:
            pass

    def send_line(self, payload: object) -> None:
        self.wfile.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    args = parse_args()
    index = FeatureIndex(args.activations or args.sae_dir / "activations")
    probe = LazyProbe(index, args.sae_dir, args.device)
    if not index.summary_path.exists():
        index.summarize(probe().sae, probe().model)
    index.load_summary()
    index.load_interpretations()

    server = ThreadingHTTPServer((args.host, args.port), partial(ExplorerHandler, index=index, probe=probe))
    print(f"Feature explorer: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

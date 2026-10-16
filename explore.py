"""Serve a local web explorer over the recorded SAE feature activations."""

import argparse
import json
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.experiment import load_sae
from src.features import FeatureIndex
from src.runtime import choose_device, load_causal_lm

STATIC_DIR = Path(__file__).parent / "explorer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.pt.")
    parser.add_argument("--activations", type=Path, default=None, help="Recorded activation directory. Default: <sae-dir>/activations.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto", help="Device for the one-time feature summary.")
    return parser.parse_args()


class ExplorerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, index: FeatureIndex, **kwargs):
        self.index = index
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
            self.send_json(self.route(url.path, query))
        except (KeyError, ValueError, IndexError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def route(self, path: str, query: dict[str, str]) -> object:
        if path == "/api/overview":
            return self.index.overview()
        if path == "/api/features":
            low, high = (float(query["low"]), float(query["high"])) if "low" in query else (None, None)
            return self.index.browse(query["sort"], int(query["offset"]), int(query["limit"]), low, high)
        if path.startswith("/api/feature/"):
            return self.index.feature(int(path.removeprefix("/api/feature/")))
        if path == "/api/token":
            return self.index.token_search(query["q"])
        raise KeyError(f"unknown endpoint {path}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def summarize(index: FeatureIndex, sae_dir: Path, device_name: str) -> None:
    device = choose_device(device_name)
    sae, config = load_sae(sae_dir, device)
    index.summarize(sae, load_causal_lm(config.model_id, config.model_dtype, device))


def main() -> None:
    args = parse_args()
    index = FeatureIndex(args.activations or args.sae_dir / "activations")
    if not index.summary_path.exists():
        summarize(index, args.sae_dir, args.device)
    index.load_summary()
    index.load_interpretations()

    server = ThreadingHTTPServer((args.host, args.port), partial(ExplorerHandler, index=index))
    print(f"Feature explorer: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

"""Name SAE features from their recorded activations, and check each name on held-out text and against the model itself."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.experiment import load_sae
from src.features import FeatureIndex
from src.interpret import (
    EXPLAIN_SCHEMA,
    GENERATE_SCHEMA,
    MIN_FIRES,
    Judge,
    Probe,
    Sampler,
    detect_prompt,
    detect_schema,
    detection_scores,
    explain_prompt,
    generate_prompt,
    generation_scores,
    measured_pairs,
    shuffled,
)
from src.runtime import choose_device, find_transformer_layers, load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output directory containing config.json and sae_final.pt.")
    parser.add_argument("--activations", type=Path, default=None, help="Recorded activation directory. Default: <sae-dir>/activations.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API of the explainer and judge model, e.g. http://127.0.0.1:8080/v1 for llama-server.")
    parser.add_argument("--model", default="judge", help="Model name sent with each request; llama-server ignores it. Default: judge.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--features", type=int, nargs="+", help=f"Features to interpret. Default: every feature with at least {MIN_FIRES} recorded activations.")
    selection.add_argument("--sample", type=int, help="Interpret this many features, spread evenly over activation density.")
    parser.add_argument("--workers", type=int, default=4, help="Features interpreted concurrently; match the server's parallel slots. Default: 4.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL; features already in it are skipped. Default: interpretations.jsonl in the activation directory.")
    return parser.parse_args()


def choose_features(index: FeatureIndex, args: argparse.Namespace) -> list[int]:
    if args.features:
        return args.features
    eligible = np.flatnonzero(index.counts >= MIN_FIRES)
    if args.sample is None:
        return eligible.tolist()
    by_density = eligible[np.argsort(index.density[eligible], kind="stable")]
    rng = np.random.default_rng(0)
    return [int(rng.choice(stratum)) for stratum in np.array_split(by_density, args.sample)]


def main() -> None:
    args = parse_args()
    activations = args.activations or args.sae_dir / "activations"
    output = args.output or activations / "interpretations.jsonl"
    index = FeatureIndex(activations)
    done = {json.loads(line)["feature"] for line in output.read_text().splitlines()} if output.exists() else set()
    features = [feature for feature in choose_features(index, args) if feature not in done]

    device = choose_device(args.device)
    sae, config = load_sae(args.sae_dir, device)
    model = load_causal_lm(config.model_id, config.model_dtype, device)
    probe = Probe(model, find_transformer_layers(model)[config.layer_index], sae, index.tokenizer, index.max, index.vocab)
    sampler = Sampler(index, sae.decoder_weight.float().cpu())
    judge = Judge(args.base_url, args.model)

    def interpret(feature: int) -> dict:
        examples = sampler.sample(feature)
        test = shuffled(examples.test, feature)
        explanation = judge(explain_prompt(sampler, feature, examples.explain), EXPLAIN_SCHEMA, max_tokens=512)
        answers = judge(detect_prompt(explanation, test), detect_schema(len(test)), max_tokens=512)["fires"]
        pairs = judge(generate_prompt(explanation), GENERATE_SCHEMA, max_tokens=2048)["pairs"]
        fires = probe.max_activations([pair["fires"] for pair in pairs], feature)
        silent = probe.max_activations([pair["silent"] for pair in pairs], feature)
        pairs = measured_pairs(pairs, fires, silent)
        return {
            "feature": feature,
            **explanation,
            "detection": detection_scores(test, answers),
            "generation": generation_scores(pairs),
            "pairs": pairs,
            "steering": probe.steer(feature),
        }

    def attempt(feature: int) -> dict | None:
        # A failed feature is reported and left out of the output, so the next run retries it.
        try:
            return interpret(feature)
        except Exception as error:
            tqdm.write(f"Feature {feature} failed: {error!r}")
            return None

    print(f"Interpreting {len(features):,} features ({len(done):,} already in {output}) with {args.base_url}")
    with output.open("a") as file, ThreadPoolExecutor(args.workers) as pool:
        for record in tqdm(pool.map(attempt, features), total=len(features), unit="feature", desc="Interpret", dynamic_ncols=True):
            if record is not None:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()


if __name__ == "__main__":
    main()

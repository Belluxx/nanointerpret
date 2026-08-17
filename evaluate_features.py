from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm

from src.interventions import InterventionGenerator, InterventionRequest
from src.runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a range of activated SAE features and rank how well the completions match their feature titles.")
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    feature_selection = parser.add_mutually_exclusive_group(required=True)
    feature_selection.add_argument("--feature-id-range", type=int, nargs=2, metavar=("START", "STOP"), help="Half-open feature range: START is included and STOP is excluded.")
    feature_selection.add_argument("--feature-activation-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="Select features with an activation count between MIN and MAX, inclusive.")
    parser.add_argument("--names", type=Path, help="Feature-title JSONL. Default: feature_names.jsonl in --sae-dir.")
    parser.add_argument("--activations", type=Path, help="Activation data directory. Default: activations in --sae-dir.")
    parser.add_argument("--strength", type=float, default=1.0, help="Clamp strength as a multiple of each feature's recorded maximum. Default: 1.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1", help="OpenAI-compatible judge URL. Default: http://127.0.0.1:9000/v1.")
    parser.add_argument("--model", default="model_default", help="Judge model name. Default: model_default.")
    parser.add_argument("--api-key", help="Judge API key. Default: OPENAI_API_KEY, or 'not-needed' if unset.")
    parser.add_argument("--judge-max-tokens", type=int, default=512, help="Judge completion-token budget, including reasoning. Default: 512.")
    return parser.parse_args()


def load_titles(path: Path) -> dict[int, str]:
    titles = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                feature_id = int(record["feature_id"])
                title = record["title"]
                if not isinstance(title, str) or not title.strip():
                    raise TypeError
                titles[feature_id] = title
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid feature title on line {line_number} of {path}"
                ) from error
    return titles


def judge_prompt(completion: str, title: str) -> str:
    return (
        f"Feature: {title}\n"
        f"Completion: {completion}\n"
        "Is the completion coherent with the feature? Answer only Yes or No."
    )


def answer_label(token: str) -> str | None:
    normalized = token.strip().strip(".,:;!?\"'`").lower()
    return normalized if normalized in {"yes", "no"} else None


def coherence_score(
    client: OpenAI,
    model: str,
    completion: str,
    title: str,
    max_tokens: int,
) -> float:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt(completion, title)}],
        temperature=0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=20,
    )
    try:
        choice = response.choices[0]
        final_label = answer_label(choice.message.content or "")
        if final_label is None:
            raise ValueError
        token_logprobs = choice.logprobs.content
        decision = next(
            token
            for token in reversed(token_logprobs)
            if answer_label(token.token) == final_label
        )
    except (AttributeError, IndexError, StopIteration, TypeError, ValueError) as error:
        raise ValueError(
            "the judge did not return a final Yes/No token with logprobs; "
            "increase --judge-max-tokens"
        ) from error

    answer_probabilities = {"yes": 0.0, "no": 0.0}
    for candidate in decision.top_logprobs or []:
        label = answer_label(candidate.token)
        if label is not None:
            answer_probabilities[label] += math.exp(float(candidate.logprob))
    if not all(answer_probabilities.values()):
        raise ValueError(
            "the judge did not include both Yes and No in its top logprobs"
        )
    yes = answer_probabilities["yes"]
    return yes / sum(answer_probabilities.values())


def main() -> None:
    args = parse_args()
    names_path = args.names or args.sae_dir / "feature_names.jsonl"
    activations_path = args.activations or args.sae_dir / "activations"
    feature_max = np.load(activations_path / "feature_max.npy", mmap_mode="r")

    if args.feature_id_range is not None:
        feature_ids = range(*args.feature_id_range)
    else:
        minimum, maximum = args.feature_activation_range
        feature_ptr = np.load(activations_path / "feature_ptr.npy", mmap_mode="r")
        activation_counts = np.diff(feature_ptr)
        feature_ids = np.flatnonzero((activation_counts >= minimum) & (activation_counts <= maximum)).tolist()

    titles = load_titles(names_path)
    missing_title = next(
        (feature_id for feature_id in feature_ids if feature_id not in titles), None
    )
    if missing_title is not None:
        raise ValueError(f"feature {missing_title} has no title in {names_path}")

    device = choose_device(args.device)
    print(f"Loading intervention model on {device} ...")
    generator = InterventionGenerator.from_sae_dir(args.sae_dir, device)
    if len(feature_max) != generator.sae.d_sae:
        raise ValueError(
            "activation data and SAE have different numbers of features: "
            f"{len(feature_max)} and {generator.sae.d_sae}"
        )

    request_args = {
        "prompt": args.prompt,
        "mode": "clamp",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    results = []
    for feature_id in tqdm(feature_ids, unit="feature", desc="Generate"):
        amount = float(feature_max[feature_id]) * args.strength
        request = InterventionRequest(
            feature_id=feature_id,
            amount=amount,
            **request_args,
        )
        completion = generator.generate_pair(request)["intervened"]
        results.append(
            {
                "feature_id": feature_id,
                "title": titles[feature_id],
                "completion": completion,
            }
        )

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
        timeout=300,
    )
    for result in tqdm(results, unit="feature", desc="Judge"):
        result["score"] = coherence_score(
            client,
            args.model,
            result["completion"],
            result["title"],
            args.judge_max_tokens,
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    print("\nRanked features:\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. Feature {result['feature_id']} | "
            f"score {result['score']:.6g} | {result['title']}"
        )
        print(result["completion"].strip() or "[empty completion]")
        print()


if __name__ == "__main__":
    main()

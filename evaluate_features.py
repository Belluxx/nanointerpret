from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm

from src.data import INTERPRETATIONS_FILENAME, load_interpretations
from src.interventions import InterventionGenerator, InterventionRequest
from src.runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score each feature based on how well they respond to clamping interventions")
    parser.add_argument("--sae-dir", type=Path, required=True, help="Training output dir containing config.json and sae_final.pt.")
    parser.add_argument("--activations", type=Path, help="Activation data dir. Default: activations in --sae-dir.")
    parser.add_argument("--interpretations", type=Path, help=f"Feature-interpretation JSONL. Default: {INTERPRETATIONS_FILENAME} in --sae-dir.")
    feature_selection = parser.add_mutually_exclusive_group()
    feature_selection.add_argument("--feature-ids", type=int, nargs="+", help="Analyze only these features. Default: every SAE feature.")
    feature_selection.add_argument("--feature-id-range", type=int, nargs=2, metavar=("START", "STOP"), help="Analyze a half-open feature range: START is included and STOP is excluded.")
    feature_selection.add_argument("--feature-activation-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="Analyze features with an activation count between MIN and MAX, inclusive.")
    parser.add_argument("--prompt", required=True, help="Text from which the LLM generates a completion while the feature is clamped.")
    parser.add_argument("--strength", type=float, default=1.0, help="Clamp strength as a multiple of each feature's recorded maximum. Default: 1.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens generated for each feature. Default: 64.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--generation-concurrent", type=int, default=64, help="Number of feature interventions generated in each model batch. Default: 64.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL. OpenAI: https://api.openai.com/v1, LMStudio: http://localhost:1234/v1, llama.cpp: http://localhost:<PORT>/v1, Ollama: http://localhost:11434/v1")
    parser.add_argument("--model", required=True, help="Judge model identifier")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--judge-concurrent", type=int, default=1, help="Maximum number of concurrent judge requests. Default: 1.")
    parser.add_argument("--output", type=Path, help="Output JSONL path. Default: feature_scores.jsonl in --sae-dir.")
    args = parser.parse_args()
    args.output = args.output or args.sae_dir / "feature_scores.jsonl"
    return args


def judge_prompt(completion: str, title: str) -> str:
    return (
        f"Title: {title}\n"
        f"Completion: {completion}\n"
        "Is the completion coherent with the title? Respond with only Yes or No. "
        "Do not reason or explain."
    )


def answer_label(token: str) -> str | None:
    normalized = token.strip().strip(".,:;!?\"'`").lower()
    return normalized if normalized in {"yes", "no"} else None


def coherence_score(
    client: OpenAI,
    model: str,
    completion: str,
    title: str,
) -> float:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt(completion, title)}],
        temperature=0,
        max_tokens=3,
        reasoning_effort="none",
        logprobs=True,
        top_logprobs=20,
    )
    try:
        choice = response.choices[0]
        decision = next(
            token
            for token in choice.logprobs.content
            if answer_label(token.token) is not None
        )
    except (AttributeError, IndexError, StopIteration, TypeError) as error:
        raise ValueError("the judge did not return Yes or No") from error

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
    no = answer_probabilities["no"]
    return yes / (yes + no)


def resume_progress(checkpoint: Path, requested: list[int]) -> list[dict]:
    if not checkpoint.exists():
        return []

    answer = input(f"{checkpoint} already exists. Continue? [Y/n] ").strip().lower()
    if answer not in {"", "y", "yes"}:
        return []

    results = []
    with checkpoint.open(encoding="utf-8") as saved_output:
        for completed, line in enumerate(saved_output, start=1):
            result = json.loads(line)
            if (
                completed > len(requested)
                or result["feature_id"] != requested[completed - 1]
            ):
                raise ValueError(
                    f"{checkpoint} does not match the requested feature order"
                )
            results.append(result)

    print(f"Continuing after {len(results):,} completed features.")
    return results


def main() -> None:
    args = parse_args()
    interpretations_path = (
        args.interpretations or args.sae_dir / INTERPRETATIONS_FILENAME
    )
    activations_path = args.activations or args.sae_dir / "activations"
    feature_max = np.load(activations_path / "feature_max.npy", mmap_mode="r")

    if args.feature_id_range is not None:
        feature_ids = range(*args.feature_id_range)
    elif args.feature_activation_range is not None:
        minimum, maximum = args.feature_activation_range
        feature_ptr = np.load(activations_path / "feature_ptr.npy", mmap_mode="r")
        activation_counts = np.diff(feature_ptr)
        feature_ids = np.flatnonzero((activation_counts >= minimum) & (activation_counts <= maximum)).tolist()
    else:
        feature_ids = args.feature_ids or range(len(feature_max))

    feature_ids = list(feature_ids)
    interpretations = load_interpretations(interpretations_path)
    missing_interpretation = next(
        (feature_id for feature_id in feature_ids if feature_id not in interpretations),
        None,
    )
    if missing_interpretation is not None:
        raise ValueError(
            f"feature {missing_interpretation} has no interpretation in "
            f"{interpretations_path}"
        )

    selected_count = len(feature_ids)
    feature_ids = [
        feature_id
        for feature_id in feature_ids
        if interpretations[feature_id]["category"] is not None
    ]
    skipped_count = selected_count - len(feature_ids)
    if skipped_count:
        print(
            f"Skipping {skipped_count:,} features with insufficient activation "
            "data or no coherent interpretation."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(args.output.suffix + ".tmp")
    results = resume_progress(checkpoint, feature_ids)
    completed = len(results)

    if completed < len(feature_ids):
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
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=300,
        )

        def judge(result: dict) -> dict:
            try:
                score = coherence_score(
                    client,
                    args.model,
                    result["completion"],
                    result["title"],
                )
                return {**result, "score": score}
            except Exception as error:
                return {
                    **result,
                    "score": None,
                    "judge_error": f"{type(error).__name__}: {error}",
                }

        with checkpoint.open(
            "a" if completed else "w", encoding="utf-8"
        ) as output, ThreadPoolExecutor(
            max_workers=args.judge_concurrent
        ) as executor, tqdm(
            total=len(feature_ids),
            initial=completed,
            unit="feature",
            desc="Evaluate",
            dynamic_ncols=True,
        ) as progress:
            for start in range(
                completed, len(feature_ids), args.generation_concurrent
            ):
                batch_ids = feature_ids[
                    start : start + args.generation_concurrent
                ]
                requests = [
                    InterventionRequest(
                        feature_id=feature_id,
                        amount=float(feature_max[feature_id])
                        * args.strength,
                        **request_args,
                    )
                    for feature_id in batch_ids
                ]
                completions = generator.generate_intervened(requests)
                generated = (
                    {
                        "feature_id": feature_id,
                        **interpretations[feature_id],
                        "completion": completion,
                    }
                    for feature_id, completion in zip(
                        batch_ids, completions
                    )
                )
                for result in executor.map(judge, generated):
                    results.append(result)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    progress.update()

    results.sort(
        key=lambda result: result["score"] if result["score"] is not None else -1.0,
        reverse=True,
    )
    ranked_temporary = args.output.with_suffix(args.output.suffix + ".ranked.tmp")
    with ranked_temporary.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    os.replace(ranked_temporary, args.output)
    checkpoint.unlink(missing_ok=True)

    print("\nRanked features:\n")
    for rank, result in enumerate(results, start=1):
        score = result["score"]
        score_text = f"{score:.6g}" if score is not None else "ERROR"
        print(
            f"{rank}. Feature {result['feature_id']} | "
            f"score {score_text} | {result['title']}"
        )
        print(result["completion"].strip() or "[empty completion]")
        if "judge_error" in result:
            print(f"Judge error: {result['judge_error']}")
        print()
    print(f"Saved {len(results):,} ranked results to {args.output}")


if __name__ == "__main__":
    main()

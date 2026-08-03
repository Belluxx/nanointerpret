from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
INSUFFICIENT_TITLE = "Insufficient activation data"
MAX_PREFIX_TOKENS = 96


@dataclass(frozen=True)
class Example:
    bucket: str
    activation: float
    percentile: float
    text: str


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask an OpenAI-compatible LLM to name SAE features.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", help="API key. Default: OPENAI_API_KEY, or 'not-needed' if unset.")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output JSONL path. Default: next to the analysis artifact.")
    parser.add_argument("--feature-ids", type=nonnegative_int, nargs="+", help="Interpret only these features. Default: every SAE feature.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable model reasoning. Reasoning is enabled by default.")
    parser.add_argument("--max-tokens", type=positive_int, help="Completion-token budget. Default: 32768, or 32 with --no-reasoning.")
    parser.add_argument("--concurrent", type=positive_int, default=1, help="Number of concurrent interpretation requests. Default: 1.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decode_marked_prefix(tokenizer, token_ids: np.ndarray) -> str:
    decode_args = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    before = tokenizer.decode(token_ids[:-1].tolist(), **decode_args)
    through = tokenizer.decode(token_ids.tolist(), **decode_args)
    target = (
        through[len(before) :]
        if through.startswith(before)
        else tokenizer.decode([int(token_ids[-1])], **decode_args)
    )
    whitespace_length = len(target) - len(target.lstrip())
    whitespace = target[:whitespace_length]
    token_text = target[whitespace_length:]
    if not token_text:
        token_text = tokenizer.convert_ids_to_tokens(int(token_ids[-1]))
    return f"{before}{whitespace}<<{token_text}>>"


def render_example(
    tokenizer,
    token_ids: np.ndarray,
    context_ptr: np.ndarray,
    row_ptr: np.ndarray,
    activation_index: int,
    activation: float,
    percentile: float,
    bucket: str,
) -> Example:
    token_position = int(np.searchsorted(row_ptr, activation_index, side="right") - 1)
    context_index = int(
        np.searchsorted(context_ptr, token_position, side="right") - 1
    )
    context_start = int(context_ptr[context_index])
    prefix_start = max(context_start, token_position - MAX_PREFIX_TOKENS + 1)
    text = decode_marked_prefix(
        tokenizer, token_ids[prefix_start : token_position + 1]
    )
    return Example(bucket, activation, percentile, text)


def choose_examples(
    feature_id: int,
    activation_indices: np.ndarray,
    values: np.ndarray,
    token_ids: np.ndarray,
    context_ptr: np.ndarray,
    row_ptr: np.ndarray,
    tokenizer,
    seed: int,
) -> list[Example] | None:
    feature_values = values[activation_indices]
    count = len(feature_values)
    if count < 10:
        return None

    ranked_indices = np.argsort(feature_values, kind="stable")
    inverse_rank = np.empty(count, dtype=np.int64)
    inverse_rank[ranked_indices] = np.arange(count)
    token_positions = np.searchsorted(
        row_ptr, activation_indices, side="right"
    ) - 1
    context_indices = np.searchsorted(
        context_ptr, token_positions, side="right"
    ) - 1
    used_activations: set[int] = set()
    used_contexts: set[int] = set()
    rng = random.Random(seed + feature_id)

    def activation_index_for_rank(rank: int) -> int:
        return int(activation_indices[int(ranked_indices[rank])])

    def context_index_for_rank(rank: int) -> int:
        return int(context_indices[int(ranked_indices[rank])])

    def make_example(rank: int, bucket: str) -> Example:
        local_index = int(ranked_indices[rank])
        percentile = 100.0 * (rank + 1) / count
        return render_example(
            tokenizer,
            token_ids,
            context_ptr,
            row_ptr,
            int(activation_indices[local_index]),
            float(feature_values[local_index]),
            percentile,
            bucket,
        )

    def select_examples(
        ranks,
        size: int,
        bucket: str,
        *,
        randomize: bool,
    ) -> list[Example] | None:
        available = [
            int(rank)
            for rank in ranks
            if activation_index_for_rank(int(rank)) not in used_activations
        ]
        selected: list[Example] = []
        for _ in range(size):
            if not available:
                return None
            unique_context = [
                rank
                for rank in available
                if context_index_for_rank(rank) not in used_contexts
            ]
            candidates = unique_context or available
            rank = rng.choice(candidates) if randomize else candidates[0]
            available.remove(rank)
            used_activations.add(activation_index_for_rank(rank))
            used_contexts.add(context_index_for_rank(rank))
            selected.append(make_example(rank, bucket))
        return selected

    top_examples = select_examples(
        range(count - 1, -1, -1), 10, "Top", randomize=False
    )
    if top_examples is None:
        return None
    examples = top_examples

    rank_percentiles = 100.0 * (np.arange(count) + 1) / count
    for lower, upper, label in (
        (25, 50, "25-50"),
        (50, 75, "50-75"),
        (75, 90, "75-90"),
        (90, 99, "90-99"),
    ):
        start = int(np.searchsorted(rank_percentiles, lower, side="left"))
        stop = int(np.searchsorted(rank_percentiles, upper, side="left"))
        if stop - start < 2:
            return None
        bucket_examples = select_examples(
            range(start, stop), 2, label, randomize=True
        )
        if bucket_examples is None:
            return None
        examples.extend(bucket_examples)

    random_ranks = (int(inverse_rank[local_index]) for local_index in range(count))
    random_examples = select_examples(
        random_ranks, 5, "Random positive", randomize=True
    )
    if random_examples is None:
        return None
    examples.extend(random_examples)
    return examples


def feature_prompt(examples: list[Example]) -> str:
    sections = [
        "Infer the feature's core concept from the examples below.\n"
        "Focus primarily on high-activation examples, but use weaker examples to "
        "detect broader meanings or polysemanticity.\n"
        "The token inside << >> is the token whose activation is reported."
    ]
    for index, example in enumerate(examples, start=1):
        sections.append(
            f"Example {index}\n"
            f"Bucket: {example.bucket}\n"
            f"Activation: {example.activation:.6g}\n"
            f"Percentile: {example.percentile:.1f}\n"
            f"Text: {example.text}"
        )
    sections.append(
        "Give this feature a very concise, specific title. "
        "Return only the plain title, with no quotes, label or explanation."
    )
    return "\n\n".join(sections)


def request_title(
    client: OpenAI,
    model: str,
    prompt: str,
    reasoning: bool = True,
    max_tokens: int | None = None,
) -> str:
    reasoning_options = {} if reasoning else {"reasoning_effort": "none"}
    max_tokens = max_tokens or (32_768 if reasoning else 32)
    for retry in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at interpreting sparse autoencoder "
                            "features."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                **reasoning_options,
            )
            content = completion.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("the model returned an empty feature title")

            title = content.strip().splitlines()[0].strip()
            if title.lower().startswith("title:"):
                title = title[6:].strip()
            title = title.strip('"\'`').strip()
            if not title:
                raise ValueError("the model returned an empty feature title")
            return title
        except Exception:
            if retry == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.analysis.with_name("feature_names.jsonl")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    with np.load(args.analysis) as analysis:
        metadata = json.loads(analysis["metadata"].item())
        token_ids = analysis["token_ids"]
        context_ptr = analysis["context_ptr"]
        row_ptr = analysis["row_ptr"]
        feature_ids = analysis["feature_ids"]
        values = analysis["values"]

        d_sae = int(metadata["d_sae"])
        requested = args.feature_ids or range(d_sae)
        invalid = [feature_id for feature_id in requested if feature_id >= d_sae]
        if invalid:
            raise ValueError(
                f"feature IDs must be below {d_sae}; got {invalid[0]}"
            )

        tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"])
        counts = np.bincount(feature_ids, minlength=d_sae)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        feature_order = np.argsort(feature_ids, kind="stable")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")

        def interpret_feature(feature_id: int) -> tuple[int, str, bool]:
            start, stop = offsets[feature_id : feature_id + 2]
            activation_indices = feature_order[start:stop]
            examples = choose_examples(
                feature_id,
                activation_indices,
                values,
                token_ids,
                context_ptr,
                row_ptr,
                tokenizer,
                args.seed,
            )
            if examples is None:
                return feature_id, INSUFFICIENT_TITLE, True
            title = request_title(
                client,
                args.model,
                feature_prompt(examples),
                reasoning=not args.no_reasoning,
                max_tokens=args.max_tokens,
            )
            return feature_id, title, False

        insufficient = 0
        with temporary.open("w", encoding="utf-8") as output, ThreadPoolExecutor(
            max_workers=args.concurrent
        ) as executor:
            results = executor.map(interpret_feature, requested)
            for feature_id, title, was_insufficient in tqdm(
                results,
                total=len(requested),
                unit="feature",
                desc="Interpret",
                dynamic_ncols=True,
            ):
                insufficient += was_insufficient
                output.write(
                    json.dumps(
                        {"feature_id": feature_id, "title": title},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
        os.replace(temporary, output_path)

    print(f"Saved {len(requested):,} feature names to {output_path}")
    if insufficient:
        print(
            f"Used '{INSUFFICIENT_TITLE}' for {insufficient:,} features "
            "without the full requested evidence set."
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer


ANALYSIS_PATH = Path("artifacts/sae_gemma_3_270m/analysis.npz")


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
    parser.add_argument("--api-key", default=None, help="API key. Default: OPENAI_API_KEY, or 'not-needed' if unset.")
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path. Default: next to the analysis artifact.")
    parser.add_argument("--feature-ids", type=nonnegative_int, nargs="+", default=None, help="Interpret only these features. Default: every SAE feature.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable model reasoning. Reasoning is enabled by default.")
    parser.add_argument("--max-tokens", type=positive_int, default=None, help="Completion-token budget. Default: 32768, or 32 with --no-reasoning.")
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
    whitespace = re.match(r"\s*", target).group()
    token_text = target[len(whitespace) :]
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
    text = decode_marked_prefix(tokenizer, token_ids[context_start : token_position + 1])
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

    ranked = np.argsort(feature_values, kind="stable")
    inverse_rank = np.empty(count, dtype=np.int64)
    inverse_rank[ranked] = np.arange(count)

    def make_example(local_index: int, bucket: str) -> Example:
        rank = int(inverse_rank[local_index])
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

    examples: list[Example] = []
    top_texts: set[str] = set()
    for local_index in ranked[::-1]:
        example = make_example(int(local_index), "Top")
        if example.text in top_texts:
            continue
        examples.append(example)
        top_texts.add(example.text)
        if len(examples) == 10:
            break
    if len(examples) < 10:
        return None

    rng = random.Random(seed + feature_id)
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
        for rank in rng.sample(range(start, stop), 2):
            examples.append(make_example(int(ranked[rank]), label))

    if count < 5:
        return None
    for local_index in rng.sample(range(count), 5):
        examples.append(make_example(local_index, "Random positive"))
    return examples


def feature_prompt(examples: list[Example]) -> str:
    sections = [
        "Infer the feature's core concept from the examples below.\n"
        "Focus primarily on high-activation examples, but use weaker examples to detect broader meanings or polysemanticity.\n"
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
        "Give this feature a concise, specific title. "
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
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at interpreting sparse autoencoder features.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        **reasoning_options,
    )
    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("the model returned an empty feature title")
    title = content.strip().splitlines()[0].strip().strip('"\'`')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    if not title:
        raise ValueError("the model returned an empty feature title")
    return title


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
        requested = args.feature_ids or list(range(d_sae))
        invalid = [feature_id for feature_id in requested if feature_id >= d_sae]
        if invalid:
            raise ValueError(
                f"feature IDs must be below {d_sae}; got {invalid[0]}"
            )

        tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"])
        counts = np.bincount(feature_ids, minlength=d_sae)
        offsets = np.empty(d_sae + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        feature_order = np.argsort(feature_ids, kind="stable")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        insufficient = 0
        with temporary.open("w", encoding="utf-8") as output:
            for feature_id in tqdm(
                requested, unit="feature", desc="Interpret", dynamic_ncols=True
            ):
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
                    title = "Insufficient activation data"
                    insufficient += 1
                else:
                    title = request_title(
                        client,
                        args.model,
                        feature_prompt(examples),
                        reasoning=not args.no_reasoning,
                        max_tokens=args.max_tokens,
                    )
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
            f"Used 'Insufficient activation data' for {insufficient:,} features "
            "without the full requested evidence set."
        )


if __name__ == "__main__":
    main()

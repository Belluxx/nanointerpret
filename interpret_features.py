from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.data import load_analysis
from src.feature_examples import (
    DEFAULT_EXAMPLE_SEED,
    choose_activation_examples,
)


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 300
INSUFFICIENT_TITLE = "Insufficient activation data"
MAX_PREFIX_TOKENS = 64
RETRYABLE_STATUS_CODES = {408, 409, 429}


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
    parser.add_argument("--max-tokens", type=positive_int, help="Completion-token budget. Default: 32768, or 64 with --no-reasoning.")
    parser.add_argument("--concurrent", type=positive_int, default=1, help="Number of concurrent interpretation requests. Default: 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_EXAMPLE_SEED)
    return parser.parse_args()


def decode_marked_prefix(tokenizer, token_ids) -> str:
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
    token_ids,
    context_size: int,
    token_position: int,
    activation: float,
    percentile: float,
    bucket: str,
) -> Example:
    context_start = token_position // context_size * context_size
    prefix_start = max(context_start, token_position - MAX_PREFIX_TOKENS + 1)
    text = decode_marked_prefix(
        tokenizer, token_ids[prefix_start : token_position + 1]
    )
    return Example(bucket, activation, percentile, text)


def choose_examples(
    feature_id: int,
    token_positions,
    values,
    token_ids,
    context_size: int,
    tokenizer,
    seed: int,
) -> list[Example] | None:
    selections = choose_activation_examples(
        feature_id,
        token_positions,
        values,
        context_size,
        seed,
    )
    if selections is None:
        return None

    return [
        render_example(
            tokenizer,
            token_ids,
            context_size,
            selection.token_position,
            selection.activation,
            selection.percentile,
            selection.bucket,
        )
        for selection in selections
    ]


def feature_prompt(examples: list[Example]) -> str:
    sections = [
        "Infer the feature's core concept from the examples below.\n"
        "Focus primarily on high-activation examples, but use weaker examples to "
        "detect broader meanings or polysemanticity.\n"
        "The token inside << >> is the token that activates the feature."
    ]
    categories = (
        ("Top activations", "Top"),
        ("Very high activations", "90-99"),
        ("High activations", "75-90"),
        ("Medium activations", "50-75"),
        ("Low activations", "25-50"),
        ("Random activations", "Random positive"),
    )
    for heading, bucket in categories:
        texts = [example.text for example in examples if example.bucket == bucket]
        if texts:
            sections.append(f"{heading}:\n" + "\n".join(f"- {text}" for text in texts))
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
    max_tokens = max_tokens or (32_768 if reasoning else 64)
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
            if not completion.choices:
                raise ValueError("the model returned no completion choices")
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
        except Exception as error:
            retryable = (
                isinstance(error, ValueError)
                or isinstance(error, APIConnectionError)
                or (
                    isinstance(error, APIStatusError)
                    and (
                        error.status_code in RETRYABLE_STATUS_CODES
                        or error.status_code >= 500
                    )
                )
            )
            if not retryable or retry == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.analysis.with_name("feature_names.jsonl")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    analysis = load_analysis(args.analysis)
    metadata = analysis.metadata
    d_sae = len(analysis.feature_ptr) - 1
    context_size = int(metadata["context_size"])
    requested = args.feature_ids or range(d_sae)
    invalid = [feature_id for feature_id in requested if feature_id >= d_sae]
    if invalid:
        raise ValueError(f"feature IDs must be below {d_sae}; got {invalid[0]}")

    tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    completed = 0
    insufficient = 0
    output_mode = "w"
    if temporary.exists():
        answer = input(f"{temporary} already exists. Continue? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            with temporary.open(encoding="utf-8") as saved_output:
                for completed, line in enumerate(saved_output, start=1):
                    record = json.loads(line)
                    if (
                        completed > len(requested)
                        or record["feature_id"] != requested[completed - 1]
                    ):
                        raise ValueError(
                            f"{temporary} does not match the requested feature order"
                        )
                    insufficient += record["title"] == INSUFFICIENT_TITLE
            output_mode = "a"
            print(f"Continuing after {completed:,} completed features.")

    remaining = requested[completed:]

    def interpret_feature(feature_id: int) -> tuple[int, str, bool]:
        start, stop = map(int, analysis.feature_ptr[feature_id : feature_id + 2])
        examples = choose_examples(
            feature_id,
            analysis.token_positions[start:stop],
            analysis.values[start:stop],
            analysis.token_ids,
            context_size,
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

    with temporary.open(output_mode, encoding="utf-8") as output, ThreadPoolExecutor(
        max_workers=args.concurrent
    ) as executor:
        results = executor.map(interpret_feature, remaining)
        for feature_id, title, was_insufficient in tqdm(
            results,
            total=len(requested),
            initial=completed,
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
            "without the complete activation evidence set."
        )


if __name__ == "__main__":
    main()

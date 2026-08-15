from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator

from openai import APIConnectionError, APIStatusError, OpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.data import load_activations


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 300
RETRYABLE_STATUS_CODES = {408, 409, 429}

INSUFFICIENT_TITLE = "Insufficient activation data"
UNCLEAR_TITLE = "No coherent interpretation"
MAX_PREFIX_TOKENS = 64
MAX_ACTIVATED_TOKENS = 5

EXAMPLES_PER_BUCKET = 5
TOP_BUCKET = "Top activations"
RANDOM_BUCKET = "Random activations"
PERCENTILE_BUCKETS = (
    ("Very high activations", 90, 99),
    ("High activations", 75, 90),
    ("Medium activations", 50, 75),
    ("Low activations", 25, 50),
)


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
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output JSONL path. Default: next to the activation data.")
    parser.add_argument("--feature-ids", type=nonnegative_int, nargs="+", help="Interpret only these features. Default: every SAE feature.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable model reasoning. Reasoning is enabled by default.")
    parser.add_argument("--max-tokens", type=positive_int, help="Completion-token budget. Default: 32768, or 64 with --no-reasoning.")
    parser.add_argument("--concurrent", type=positive_int, default=1, help="Number of concurrent interpretation requests. Default: 1.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decode_prefix(tokenizer, token_ids) -> tuple[str, str]:
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
    token_text = target.lstrip()
    if not token_text:
        token_text = tokenizer.convert_ids_to_tokens(int(token_ids[-1]))
    return through, token_text


def render_example(
    tokenizer,
    token_ids,
    token_positions,
    values,
    context_size: int,
    token_position: int,
) -> str:
    context_start = token_position // context_size * context_size
    prefix_start = max(context_start, token_position - MAX_PREFIX_TOKENS + 1)
    start = int(token_positions.searchsorted(prefix_start))
    stop = int(token_positions.searchsorted(token_position, side="right"))
    activations = [
        (int(position), float(strength))
        for position, strength in zip(token_positions[start:stop], values[start:stop])
    ]
    if len(activations) > MAX_ACTIVATED_TOKENS:
        focal = activations.pop()
        activations.sort(key=lambda item: item[1], reverse=True)
        activations = activations[: MAX_ACTIVATED_TOKENS - 1] + [focal]
        activations.sort()

    decoded = [
        (
            *decode_prefix(tokenizer, token_ids[prefix_start : position + 1]),
            strength,
        )
        for position, strength in activations
    ]
    context = decoded[-1][0]
    activation_text = "Activated tokens:\n" + "\n".join(
        f"- `{token}`: {strength:.2f}" for _, token, strength in decoded
    )
    return f"Context: {context.strip()}\n{activation_text}"


def choose_examples(
    feature_id: int,
    token_positions,
    values,
    token_ids,
    context_size: int,
    tokenizer,
    seed: int,
) -> dict[str, list[str]] | None:
    count = len(values)
    ranked = values.argsort(kind="stable")
    context_ids = token_positions // context_size
    used_positions: set[int] = set()
    used_contexts: set[int] = set()
    rng = random.Random(seed + feature_id)

    def available(rank: int, require_new_context: bool) -> bool:
        local_index = int(ranked[rank])
        return int(token_positions[local_index]) not in used_positions and (
            not require_new_context
            or int(context_ids[local_index]) not in used_contexts
        )

    def find(
        ranks: range, randomize: bool, require_new_context: bool
    ) -> int | None:
        if randomize:
            for _ in range(64):
                rank = rng.choice(ranks)
                if available(rank, require_new_context):
                    return rank
        return next(
            (rank for rank in ranks if available(rank, require_new_context)),
            None,
        )

    def select(ranks: range, *, randomize: bool) -> list[str] | None:
        if len(ranks) < EXAMPLES_PER_BUCKET:
            return None
        examples = []
        for _ in range(EXAMPLES_PER_BUCKET):
            rank = find(ranks, randomize, require_new_context=True)
            if rank is None:
                rank = find(ranks, randomize, require_new_context=False)
            if rank is None:
                return None

            local_index = int(ranked[rank])
            token_position = int(token_positions[local_index])
            used_positions.add(token_position)
            used_contexts.add(int(context_ids[local_index]))
            examples.append(
                render_example(
                    tokenizer,
                    token_ids,
                    token_positions,
                    values,
                    context_size,
                    token_position,
                )
            )
        return examples

    top = select(range(count - 1, -1, -1), randomize=False)
    if top is None:
        return None
    examples = {TOP_BUCKET: top}

    for bucket, lower, upper in PERCENTILE_BUCKETS:
        start = math.ceil(lower * count / 100) - 1
        stop = math.ceil(upper * count / 100) - 1
        selected = select(range(start, stop), randomize=True)
        if selected is None:
            return None
        examples[bucket] = selected

    selected = select(range(count), randomize=True)
    if selected is None:
        return None
    examples[RANDOM_BUCKET] = selected
    return examples


def feature_prompt(examples: dict[str, list[str]]) -> str:
    sections = [
        "Infer the feature's core concept from the examples below.\n"
        "Focus primarily on high-activation examples, but use weaker examples to detect broader meanings or polysemanticity."
    ]
    for heading, texts in examples.items():
        sections.append(
            f"## {heading}:\n"
            + "\n".join(f"```\n{text.strip()}\n```" for text in texts)
        )
    sections.append(
        f"If the examples do not support one coherent concept, return exactly \"{UNCLEAR_TITLE}\". "
        "Give this feature a very concise, specific title. "
        "Return only the plain title, with no quotes, label or explanation."
    )
    return "\n\n".join(sections)


def clean_title(content: str | None) -> str:
    if not content or not content.strip():
        raise ValueError("the model returned an empty feature title")

    title = content.strip().splitlines()[0].strip()
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    title = title.strip('"\'`').strip()
    if not title:
        raise ValueError("the model returned an empty feature title")
    return title


def request_title(
    client: OpenAI,
    model: str,
    prompt: str,
    reasoning: bool = True,
    max_tokens: int | None = None,
) -> str:
    reasoning_options = {} if reasoning else {"reasoning_effort": "none"}
    if max_tokens is None:
        max_tokens = 32_768 if reasoning else 64

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
            return clean_title(completion.choices[0].message.content)
        except Exception as error:
            retryable = (
                isinstance(error, (ValueError, APIConnectionError))
                or (
                    isinstance(error, APIStatusError)
                    and (
                        error.status_code in RETRYABLE_STATUS_CODES
                        or error.status_code >= 500
                    )
                )
            )
            if not retryable:
                raise
            if retry == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def map_bounded(
    executor: ThreadPoolExecutor,
    function: Callable[[int], tuple[int, str]],
    items: Iterable[int],
    max_pending: int,
) -> Iterator[tuple[int, str]]:
    items = iter(items)
    pending = deque()
    for item in items:
        pending.append(executor.submit(function, item))
        if len(pending) == max_pending:
            break

    while pending:
        yield pending.popleft().result()
        try:
            item = next(items)
        except StopIteration:
            continue
        pending.append(executor.submit(function, item))


def resume_progress(temporary: Path, requested: list[int]) -> tuple[int, int]:
    if not temporary.exists():
        return 0, 0

    answer = input(f"{temporary} already exists. Continue? [Y/n] ").strip().lower()
    if answer not in {"", "y", "yes"}:
        return 0, 0

    completed = 0
    insufficient = 0
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

    print(f"Continuing after {completed:,} completed features.")
    return completed, insufficient


def main() -> None:
    args = parse_args()
    output_path = args.output or args.activations.with_name("feature_names.jsonl")
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    activations = load_activations(args.activations)
    d_sae = len(activations.feature_ptr) - 1
    context_size = int(activations.metadata["context_size"])
    requested = list(args.feature_ids or range(d_sae))
    invalid = next(
        (feature_id for feature_id in requested if feature_id >= d_sae), None
    )
    if invalid is not None:
        raise ValueError(f"feature IDs must be below {d_sae}; got {invalid}")

    tokenizer = AutoTokenizer.from_pretrained(activations.metadata["model_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    completed, insufficient = resume_progress(temporary, requested)

    def interpret_feature(feature_id: int) -> tuple[int, str]:
        start, stop = map(
            int, activations.feature_ptr[feature_id : feature_id + 2]
        )
        examples = choose_examples(
            feature_id,
            activations.token_positions[start:stop],
            activations.values[start:stop],
            activations.token_ids,
            context_size,
            tokenizer,
            args.seed,
        )
        if examples is None:
            return feature_id, INSUFFICIENT_TITLE
        try:
            title = request_title(
                client,
                args.model,
                feature_prompt(examples),
                reasoning=not args.no_reasoning,
                max_tokens=args.max_tokens,
            )
        except Exception as error:
            raise RuntimeError(f"failed to interpret feature {feature_id}") from error
        return feature_id, title

    executor = ThreadPoolExecutor(max_workers=args.concurrent)
    try:
        with temporary.open("a" if completed else "w", encoding="utf-8") as output:
            results = map_bounded(
                executor,
                interpret_feature,
                requested[completed:],
                args.concurrent,
            )
            for feature_id, title in tqdm(
                results,
                total=len(requested),
                initial=completed,
                unit="feature",
                desc="Interpret",
                dynamic_ncols=True,
            ):
                insufficient += title == INSUFFICIENT_TITLE
                record = {"feature_id": feature_id, "title": title}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()
    os.replace(temporary, output_path)

    print(f"Saved {len(requested):,} feature names to {output_path}")
    if insufficient:
        print(
            f"Used '{INSUFFICIENT_TITLE}' for {insufficient:,} features "
            "without the complete activation evidence set."
        )


if __name__ == "__main__":
    main()

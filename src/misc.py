from __future__ import annotations

import re
from pathlib import Path


def compact_token_count(token_count: int) -> str:
    units = ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k"))
    for divisor, suffix in units:
        if token_count % divisor == 0:
            return f"{token_count // divisor}{suffix}"
    return str(token_count)


def model_name_and_size(model_id: str) -> tuple[str, str]:
    model_id_name = model_id.rstrip("/").rsplit("/", 1)[-1].lower()
    size_match = re.search(
        r"(?<![a-z0-9])(\d+(?:\.\d+)?[bmk])(?:$|[^a-z0-9])", model_id_name
    )
    if size_match is None:
        raise ValueError(
            f"model ID must include a size such as '-1B' or '-270M': {model_id}"
        )
    model_name = re.sub(
        r"[^a-z0-9]+", "_", model_id_name[: size_match.start()]
    ).strip("_")
    return model_name, size_match.group(1)


def experiment_output_dir(
    model_id: str,
    layer_index: int,
    width_multiplier: int,
    k: int,
    train_tokens: int,
) -> Path:
    model_name, model_size = model_name_and_size(model_id)
    run_name = (
        f"{model_name}_{model_size}_l{layer_index}_w{width_multiplier}_k{k}_"
        f"{compact_token_count(train_tokens)}"
    )
    return Path("artifacts") / run_name

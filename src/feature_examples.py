from __future__ import annotations

import math
import random

import numpy as np


DEFAULT_EXAMPLE_SEED = 42
EXAMPLES_PER_BUCKET = 5
PERCENTILE_BUCKETS = (
    (25, 50, "25-50"),
    (50, 75, "50-75"),
    (75, 90, "75-90"),
    (90, 99, "90-99"),
)
EXAMPLE_COUNT = (len(PERCENTILE_BUCKETS) + 2) * EXAMPLES_PER_BUCKET


def choose_activation_examples(
    feature_id: int,
    token_positions: np.ndarray,
    values: np.ndarray,
    context_size: int,
    seed: int = DEFAULT_EXAMPLE_SEED,
) -> dict[str, list[int]] | None:
    count = len(values)
    if count < EXAMPLE_COUNT:
        return None

    ranked = np.argsort(values, kind="stable")
    context_ids = token_positions // context_size
    used_ranks: set[int] = set()
    used_contexts: set[int] = set()
    rng = random.Random(seed + feature_id)

    def select(ranks: range, randomize: bool) -> list[int] | None:
        def find(require_new_context: bool) -> int | None:
            def available(rank: int) -> bool:
                local_index = int(ranked[rank])
                return rank not in used_ranks and (
                    not require_new_context
                    or int(context_ids[local_index]) not in used_contexts
                )

            if randomize:
                for _ in range(64):
                    rank = rng.choice(ranks)
                    if available(rank):
                        return rank
            return next((rank for rank in ranks if available(rank)), None)

        positions = []
        for _ in range(EXAMPLES_PER_BUCKET):
            rank = find(True)
            if rank is None:
                rank = find(False)
            if rank is None:
                return None

            local_index = int(ranked[rank])
            used_ranks.add(rank)
            used_contexts.add(int(context_ids[local_index]))
            positions.append(int(token_positions[local_index]))
        return positions

    top = select(range(count - 1, -1, -1), False)
    if top is None:
        return None
    examples = {"Top": top}

    for lower, upper, bucket in PERCENTILE_BUCKETS:
        start = math.ceil(lower * count / 100) - 1
        stop = math.ceil(upper * count / 100) - 1
        selected = select(range(start, stop), True)
        if selected is None:
            return None
        examples[bucket] = selected

    selected = select(range(count), True)
    if selected is None:
        return None
    examples["Random positive"] = selected
    return examples

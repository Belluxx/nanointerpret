from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


DEFAULT_EXAMPLE_SEED = 42
TOP_EXAMPLE_COUNT = 5
STRATIFIED_EXAMPLES_PER_BUCKET = 5
RANDOM_EXAMPLE_COUNT = 5
STRATIFIED_BUCKETS = (
    (25, 50, "25-50"),
    (50, 75, "50-75"),
    (75, 90, "75-90"),
    (90, 99, "90-99"),
)
COMPLETE_EXAMPLE_COUNT = (
    TOP_EXAMPLE_COUNT
    + len(STRATIFIED_BUCKETS) * STRATIFIED_EXAMPLES_PER_BUCKET
    + RANDOM_EXAMPLE_COUNT
)
RANDOM_SELECTION_ATTEMPTS = 64


@dataclass(frozen=True)
class ActivationExample:
    token_position: int
    activation: float
    percentile: float
    bucket: str


def choose_activation_examples(
    feature_id: int,
    token_positions: np.ndarray,
    values: np.ndarray,
    context_size: int,
    seed: int = DEFAULT_EXAMPLE_SEED,
) -> list[ActivationExample] | None:
    count = len(values)
    if count < COMPLETE_EXAMPLE_COUNT:
        return None

    ranked_indices = np.argsort(values, kind="stable")
    context_indices = token_positions // context_size
    used_positions: set[int] = set()
    used_contexts: set[int] = set()
    rng = random.Random(seed + feature_id)

    def token_position_for_rank(rank: int) -> int:
        return int(token_positions[int(ranked_indices[rank])])

    def context_index_for_rank(rank: int) -> int:
        return int(context_indices[int(ranked_indices[rank])])

    def make_example(rank: int, bucket: str) -> ActivationExample:
        local_index = int(ranked_indices[rank])
        return ActivationExample(
            token_position=int(token_positions[local_index]),
            activation=float(values[local_index]),
            percentile=100.0 * (rank + 1) / count,
            bucket=bucket,
        )

    def select_examples(
        ranks,
        size: int,
        bucket: str,
        *,
        randomize: bool,
    ) -> list[ActivationExample] | None:
        def is_available(rank: int, require_new_context: bool) -> bool:
            return (
                token_position_for_rank(rank) not in used_positions
                and (
                    not require_new_context
                    or context_index_for_rank(rank) not in used_contexts
                )
            )

        def find_rank(require_new_context: bool) -> int | None:
            if not randomize:
                return next(
                    (
                        int(rank)
                        for rank in ranks
                        if is_available(int(rank), require_new_context)
                    ),
                    None,
                )

            for _ in range(RANDOM_SELECTION_ATTEMPTS):
                rank = int(ranks[rng.randrange(len(ranks))])
                if is_available(rank, require_new_context):
                    return rank

            # Collisions are normally rare. If they are not, use reservoir
            # sampling to choose uniformly without building a large list.
            selected_rank = None
            candidate_count = 0
            for rank in ranks:
                rank = int(rank)
                if is_available(rank, require_new_context):
                    candidate_count += 1
                    if rng.randrange(candidate_count) == 0:
                        selected_rank = rank
            return selected_rank

        selected = []
        for _ in range(size):
            rank = find_rank(require_new_context=True)
            if rank is None:
                rank = find_rank(require_new_context=False)
            if rank is None:
                return None
            used_positions.add(token_position_for_rank(rank))
            used_contexts.add(context_index_for_rank(rank))
            selected.append(make_example(rank, bucket))
        return selected

    examples = select_examples(
        range(count - 1, -1, -1), TOP_EXAMPLE_COUNT, "Top", randomize=False
    )
    if examples is None:
        return None

    for lower, upper, label in STRATIFIED_BUCKETS:
        start = math.ceil(lower * count / 100) - 1
        stop = math.ceil(upper * count / 100) - 1
        bucket_examples = select_examples(
            range(start, stop),
            STRATIFIED_EXAMPLES_PER_BUCKET,
            label,
            randomize=True,
        )
        if bucket_examples is None:
            return None
        examples.extend(bucket_examples)

    random_examples = select_examples(
        range(count), RANDOM_EXAMPLE_COUNT, "Random positive", randomize=True
    )
    if random_examples is None:
        return None
    examples.extend(random_examples)
    return examples

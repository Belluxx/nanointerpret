from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


DEFAULT_EXAMPLE_SEED = 42
TOP_EXAMPLE_COUNT = 10
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
) -> list[ActivationExample]:
    count = len(values)
    if count == 0:
        return []

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
    ) -> list[ActivationExample]:
        available = [
            int(rank)
            for rank in ranks
            if token_position_for_rank(int(rank)) not in used_positions
        ]
        selected = []
        while available and len(selected) < size:
            unique_context = [
                rank
                for rank in available
                if context_index_for_rank(rank) not in used_contexts
            ]
            candidates = unique_context or available
            rank = rng.choice(candidates) if randomize else candidates[0]
            available.remove(rank)
            used_positions.add(token_position_for_rank(rank))
            used_contexts.add(context_index_for_rank(rank))
            selected.append(make_example(rank, bucket))
        return selected

    examples = select_examples(
        range(count - 1, -1, -1), TOP_EXAMPLE_COUNT, "Top", randomize=False
    )

    for lower, upper, label in STRATIFIED_BUCKETS:
        start = math.ceil(lower * count / 100) - 1
        stop = math.ceil(upper * count / 100) - 1
        examples.extend(
            select_examples(
                range(start, stop),
                STRATIFIED_EXAMPLES_PER_BUCKET,
                label,
                randomize=True,
            )
        )

    examples.extend(
        select_examples(
            range(count), RANDOM_EXAMPLE_COUNT, "Random positive", randomize=True
        )
    )
    return examples

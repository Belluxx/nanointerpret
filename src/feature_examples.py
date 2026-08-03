from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


DEFAULT_EXAMPLE_SEED = 42
STRATIFIED_EXAMPLES_PER_BUCKET = 4
STRATIFIED_BUCKETS = (
    (25, 50, "25-50"),
    (50, 75, "50-75"),
    (75, 90, "75-90"),
    (90, 99, "90-99"),
)


@dataclass(frozen=True)
class ActivationExample:
    activation_index: int
    activation: float
    percentile: float
    bucket: str


def choose_activation_examples(
    feature_id: int,
    activation_indices: np.ndarray,
    values: np.ndarray,
    context_ptr: np.ndarray,
    row_ptr: np.ndarray,
    seed: int = DEFAULT_EXAMPLE_SEED,
) -> list[ActivationExample] | None:
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

    def make_example(rank: int, bucket: str) -> ActivationExample:
        local_index = int(ranked_indices[rank])
        return ActivationExample(
            activation_index=int(activation_indices[local_index]),
            activation=float(feature_values[local_index]),
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
        available = [
            int(rank)
            for rank in ranks
            if activation_index_for_rank(int(rank)) not in used_activations
        ]
        selected = []
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
    for lower, upper, label in STRATIFIED_BUCKETS:
        start = int(np.searchsorted(rank_percentiles, lower, side="left"))
        stop = int(np.searchsorted(rank_percentiles, upper, side="left"))
        if stop - start < STRATIFIED_EXAMPLES_PER_BUCKET:
            return None
        bucket_examples = select_examples(
            range(start, stop),
            STRATIFIED_EXAMPLES_PER_BUCKET,
            label,
            randomize=True,
        )
        if bucket_examples is None:
            return None
        examples.extend(bucket_examples)

    random_ranks = (int(inverse_rank[index]) for index in range(count))
    random_examples = select_examples(
        random_ranks, 5, "Random positive", randomize=True
    )
    if random_examples is None:
        return None
    examples.extend(random_examples)
    return examples

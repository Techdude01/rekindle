"""Exact-catalog Recall@K for the persisted non-learned baselines."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.retrieval.sequences import SequenceExamples

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BaselineMetric:
    """One exact-catalog baseline result on a fixed replay subset."""

    name: str
    recall_at_100: float
    evaluated_examples: int

    def to_dict(self) -> dict[str, str | float | int]:
        """Return a JSON-ready representation."""
        return asdict(self)


def popularity_recall_at_k(
    model: TimeDecayedPopularity,
    examples: SequenceExamples,
    example_indices: np.ndarray,
    k: int,
    progress_callback: ProgressCallback | None = None,
) -> float:
    """Score static time-decayed popularity after filtering all prior products."""
    item_to_index = {item_id: index for index, item_id in enumerate(examples.item_ids)}
    ranked_items = [
        item_to_index[item_id]
        for item_id in sorted(model.scores, key=lambda item_id: (-model.scores[item_id], item_id))
        if item_id in item_to_index
    ]
    return _recall_from_ranker(
        examples,
        example_indices,
        k,
        lambda history, seen: _first_unseen(ranked_items, seen, k),
        progress_callback,
        "Popularity",
    )


def item_cf_recall_at_k(
    model: ItemItemCosine,
    examples: SequenceExamples,
    example_indices: np.ndarray,
    k: int,
    progress_callback: ProgressCallback | None = None,
) -> float:
    """Score sparse item-item CF after filtering all prior products."""
    return _recall_from_ranker(
        examples,
        example_indices,
        k,
        lambda history, seen: [
            model.item_to_index[item_id]
            for item_id in model.recommend(
                history_item_ids=[examples.item_ids[item_index] for item_index in history],
                seen_item_ids={examples.item_ids[item_index] for item_index in seen},
                limit=k,
            )
        ],
        progress_callback,
        "Item-item CF",
    )


def _recall_from_ranker(
    examples: SequenceExamples,
    example_indices: np.ndarray,
    k: int,
    recommend: Callable[[list[int], set[int]], list[int]],
    progress_callback: ProgressCallback | None,
    label: str,
) -> float:
    """Replay a fixed event subset using only the history available at each target."""
    if len(example_indices) == 0:
        return 0.0
    hits = 0
    report_every = 100
    for count, example_index in enumerate(example_indices, start=1):
        user_index = int(examples.user_indices[example_index])
        position = int(examples.prior_event_counts[example_index])
        history = examples.user_item_sequences[user_index][:position]
        seen = set(history)
        target = int(examples.targets[example_index])
        hits += target in recommend(history, seen)
        if progress_callback is not None and (
            count % report_every == 0 or count == len(example_indices)
        ):
            progress_callback(
                f"{label} | event {count:,}/{len(example_indices):,} "
                f"({count / len(example_indices):.0%})."
            )
    return hits / len(example_indices)


def _first_unseen(ranked_items: list[int], seen: set[int], limit: int) -> list[int]:
    """Keep a global rank ordering while removing products in the local history."""
    recommendations: list[int] = []
    for item_index in ranked_items:
        if item_index not in seen:
            recommendations.append(item_index)
            if len(recommendations) == limit:
                break
    return recommendations

"""Measure complementary coverage from multiple time-safe candidate generators."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.evaluation.baselines import _first_unseen
from rekindle.retrieval.sequences import SequenceExamples

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CandidateUnionMetrics:
    """Per-channel and combined retrieval coverage for a fixed event subset."""

    neural_recall_at_200: float
    popularity_recall_at_200: float
    item_cf_recall_at_200: float
    union_candidate_recall: float
    average_unique_candidates: float
    evaluated_examples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready result."""
        return asdict(self)


def evaluate_candidate_union(
    examples: SequenceExamples,
    example_indices: np.ndarray,
    neural_candidates: np.ndarray,
    popularity: TimeDecayedPopularity,
    item_cf: ItemItemCosine,
    candidate_count: int,
    progress_callback: ProgressCallback | None = None,
) -> CandidateUnionMetrics:
    """Evaluate three unseen-item candidate channels and their deduplicated union."""
    if neural_candidates.shape != (len(example_indices), candidate_count):
        raise ValueError("Neural candidates must have one top-K row per evaluated event")
    item_to_index = {item_id: index for index, item_id in enumerate(examples.item_ids)}
    popularity_ranked = [
        item_to_index[item_id]
        for item_id in sorted(
            popularity.scores, key=lambda item_id: (-popularity.scores[item_id], item_id)
        )
        if item_id in item_to_index
    ]
    neural_hits = popularity_hits = item_cf_hits = union_hits = 0
    unique_candidate_counts: list[int] = []
    for count, example_index in enumerate(example_indices, start=1):
        user_index = int(examples.user_indices[example_index])
        position = int(examples.prior_event_counts[example_index])
        history = examples.user_item_sequences[user_index][:position]
        seen = set(history)
        target = int(examples.targets[example_index])
        neural = neural_candidates[count - 1].tolist()
        popularity_candidates = _first_unseen(popularity_ranked, seen, candidate_count)
        item_cf_candidates = [
            item_cf.item_to_index[item_id]
            for item_id in item_cf.recommend(
                history_item_ids=[examples.item_ids[item_index] for item_index in history],
                seen_item_ids={examples.item_ids[item_index] for item_index in seen},
                limit=candidate_count,
            )
        ]
        union = set(neural).union(popularity_candidates, item_cf_candidates)
        neural_hits += target in neural
        popularity_hits += target in popularity_candidates
        item_cf_hits += target in item_cf_candidates
        union_hits += target in union
        unique_candidate_counts.append(len(union))
        if progress_callback is not None and (count % 100 == 0 or count == len(example_indices)):
            progress_callback(
                f"Candidate union | event {count:,}/{len(example_indices):,} "
                f"({count / len(example_indices):.0%})."
            )
    total = len(example_indices)
    if total == 0:
        raise ValueError("At least one evaluation example is required")
    return CandidateUnionMetrics(
        neural_recall_at_200=neural_hits / total,
        popularity_recall_at_200=popularity_hits / total,
        item_cf_recall_at_200=item_cf_hits / total,
        union_candidate_recall=union_hits / total,
        average_unique_candidates=float(np.mean(unique_candidate_counts)),
        evaluated_examples=total,
    )

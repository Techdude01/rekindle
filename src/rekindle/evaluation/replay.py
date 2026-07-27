"""Frozen-model sequential replay for the final untouched test period."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.evaluation.baselines import _first_unseen
from rekindle.evaluation.bootstrap import bootstrap_mean_confidence_intervals
from rekindle.ranking.crossfit import _affinity, _inverse_rank, _merge_source_ranks
from rekindle.retrieval.inference import load_retriever, retrieve_top_k
from rekindle.retrieval.model import select_device
from rekindle.retrieval.sequences import (
    SequenceExamples,
    build_sequence_examples,
    load_sequence_events,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ReplayMetrics:
    """Frozen-model quality for every eligible warm test event."""

    evaluated_examples: int
    neural_recall_at_50: float
    neural_recall_at_100: float
    neural_recall_at_200: float
    candidate_union_recall: float
    neural_only_hit_rate_at_10: float
    neural_only_ndcg_at_10: float
    popularity_hit_rate_at_10: float
    popularity_ndcg_at_10: float
    item_cf_hit_rate_at_10: float
    item_cf_ndcg_at_10: float
    end_to_end_hit_rate_at_10: float
    end_to_end_ndcg_at_10: float
    end_to_end_ndcg_at_10_ci95_low: float
    end_to_end_ndcg_at_10_ci95_high: float
    end_to_end_ndcg_minus_popularity_ci95_low: float
    end_to_end_ndcg_minus_popularity_ci95_high: float
    end_to_end_ndcg_minus_item_cf_ci95_low: float
    end_to_end_ndcg_minus_item_cf_ci95_high: float
    average_unique_candidates: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready result."""
        return asdict(self)


def run_test_replay(
    project_root: Path,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> ReplayMetrics:
    """Score test events with fixed global models and only their preceding histories."""
    examples = _load_test_examples(project_root, config)
    item_metadata = _load_item_metadata(project_root / config["data"]["prepared_items_path"])
    candidate_count = config["retrieval"]["candidate_count"]
    device = select_device()
    _report(progress_callback, f"Loading frozen history-only retriever on {device.type}.")
    retriever = load_retriever(project_root / "artifacts/retriever-history-only/model.pt", device)
    neural_candidates = retrieve_top_k(
        retriever,
        examples,
        np.arange(examples.count, dtype=np.int64),
        k=candidate_count,
        batch_size=config["retrieval"]["evaluation_batch_size"],
        device=device,
        progress_callback=progress_callback,
    )
    popularity = TimeDecayedPopularity.load(
        project_root / "artifacts/baselines/popularity-30d.json"
    )
    item_cf = ItemItemCosine.load(project_root / "artifacts/baselines/item-cosine")
    ranker = lgb.Booster(model_file=str(project_root / "artifacts/ranker/model.txt"))
    metrics = _score_examples(
        examples,
        neural_candidates,
        popularity,
        item_cf,
        ranker,
        item_metadata,
        candidate_count,
        bootstrap_resamples=config["evaluation"]["bootstrap_resamples"],
        confidence_level=config["evaluation"]["bootstrap_confidence_level"],
        seed=config["project"]["seed"] + 4,
        progress_callback=progress_callback,
    )
    output_path = project_root / "artifacts/evaluation/test-replay.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n", encoding="utf-8")
    return metrics


def _load_test_examples(project_root: Path, config: dict) -> SequenceExamples:
    """Build test targets with training/validation/test history but global-train vocabulary."""
    events = load_sequence_events(
        project_root / config["data"]["prepared_interactions_path"], target_split="test"
    )
    return build_sequence_examples(
        events,
        target_split="test",
        history_size=config["split"]["replay_history_size"],
        min_prior_interactions=config["split"]["min_prior_interactions"],
    )


def _score_examples(
    examples: SequenceExamples,
    neural_candidates: np.ndarray,
    popularity: TimeDecayedPopularity,
    item_cf: ItemItemCosine,
    ranker: lgb.Booster,
    metadata: dict[str, tuple[str, str]],
    candidate_count: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
    progress_callback: ProgressCallback | None,
) -> ReplayMetrics:
    """Compute source recall plus ranker NDCG with zeros for naturally missed targets."""
    if neural_candidates.shape != (examples.count, candidate_count):
        raise ValueError("Neural candidates must contain one top-K row for every test example")
    item_to_index = {item_id: index for index, item_id in enumerate(examples.item_ids)}
    popularity_ranked = [
        item_to_index[item_id]
        for item_id in sorted(
            popularity.scores, key=lambda item_id: (-popularity.scores[item_id], item_id)
        )
        if item_id in item_to_index
    ]
    neural_hits = {50: 0, 100: 0, 200: 0}
    union_hits = ranker_hits = 0
    ranker_ndcg_total = 0.0
    neural_hits_at_10 = neural_ndcg_total = 0.0
    popularity_hits_at_10 = popularity_ndcg_total = 0.0
    item_cf_hits_at_10 = item_cf_ndcg_total = 0.0
    ranker_ndcg_by_example = np.zeros(examples.count, dtype=np.float64)
    popularity_ndcg_by_example = np.zeros(examples.count, dtype=np.float64)
    item_cf_ndcg_by_example = np.zeros(examples.count, dtype=np.float64)
    candidate_counts: list[int] = []
    for example_index in range(examples.count):
        user_index = int(examples.user_indices[example_index])
        position = int(examples.prior_event_counts[example_index])
        history = examples.user_item_sequences[user_index][:position]
        seen = set(history)
        target = int(examples.targets[example_index])
        neural = neural_candidates[example_index].tolist()
        popular = _first_unseen(popularity_ranked, seen, candidate_count)
        collaborative = [
            item_cf.item_to_index[item_id]
            for item_id in item_cf.recommend(
                history_item_ids=[examples.item_ids[item_index] for item_index in history],
                seen_item_ids={examples.item_ids[item_index] for item_index in seen},
                limit=candidate_count,
            )
        ]
        neural_hit, neural_ndcg = _top_k_metrics(target, neural, k=10)
        popularity_hit, popularity_ndcg = _top_k_metrics(target, popular, k=10)
        item_cf_hit, item_cf_ndcg = _top_k_metrics(target, collaborative, k=10)
        neural_hits_at_10 += neural_hit
        neural_ndcg_total += neural_ndcg
        popularity_hits_at_10 += popularity_hit
        popularity_ndcg_total += popularity_ndcg
        item_cf_hits_at_10 += item_cf_hit
        item_cf_ndcg_total += item_cf_ndcg
        popularity_ndcg_by_example[example_index] = popularity_ndcg
        item_cf_ndcg_by_example[example_index] = item_cf_ndcg
        for cutoff in neural_hits:
            neural_hits[cutoff] += target in neural[:cutoff]
        source_ranks = _merge_source_ranks(neural, popular, collaborative)
        union_hits += target in source_ranks
        candidate_counts.append(len(source_ranks))
        if target in source_ranks:
            candidate_indices = list(source_ranks)
            scores = ranker.predict(
                _ranker_features(
                    candidate_indices,
                    source_ranks,
                    examples.item_ids,
                    history,
                    position,
                    metadata,
                    popularity,
                )
            )
            target_position = candidate_indices.index(target)
            ordered_positions = np.argsort(-scores, kind="stable")
            rank = int(np.flatnonzero(ordered_positions == target_position)[0]) + 1
            if rank <= 10:
                ranker_hits += 1
                ranker_ndcg = 1.0 / math.log2(rank + 1)
                ranker_ndcg_total += ranker_ndcg
                ranker_ndcg_by_example[example_index] = ranker_ndcg
        completed = example_index + 1
        if progress_callback is not None and (completed % 100 == 0 or completed == examples.count):
            progress_callback(
                f"Test replay | event {completed:,}/{examples.count:,} "
                f"({completed / examples.count:.0%})."
            )
    intervals = bootstrap_mean_confidence_intervals(
        {
            "end_to_end": ranker_ndcg_by_example,
            "end_to_end_minus_popularity": ranker_ndcg_by_example - popularity_ndcg_by_example,
            "end_to_end_minus_item_cf": ranker_ndcg_by_example - item_cf_ndcg_by_example,
        },
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return ReplayMetrics(
        evaluated_examples=examples.count,
        neural_recall_at_50=neural_hits[50] / examples.count,
        neural_recall_at_100=neural_hits[100] / examples.count,
        neural_recall_at_200=neural_hits[200] / examples.count,
        candidate_union_recall=union_hits / examples.count,
        neural_only_hit_rate_at_10=neural_hits_at_10 / examples.count,
        neural_only_ndcg_at_10=neural_ndcg_total / examples.count,
        popularity_hit_rate_at_10=popularity_hits_at_10 / examples.count,
        popularity_ndcg_at_10=popularity_ndcg_total / examples.count,
        item_cf_hit_rate_at_10=item_cf_hits_at_10 / examples.count,
        item_cf_ndcg_at_10=item_cf_ndcg_total / examples.count,
        end_to_end_hit_rate_at_10=ranker_hits / examples.count,
        end_to_end_ndcg_at_10=ranker_ndcg_total / examples.count,
        end_to_end_ndcg_at_10_ci95_low=intervals["end_to_end"][0],
        end_to_end_ndcg_at_10_ci95_high=intervals["end_to_end"][1],
        end_to_end_ndcg_minus_popularity_ci95_low=intervals["end_to_end_minus_popularity"][0],
        end_to_end_ndcg_minus_popularity_ci95_high=intervals["end_to_end_minus_popularity"][1],
        end_to_end_ndcg_minus_item_cf_ci95_low=intervals["end_to_end_minus_item_cf"][0],
        end_to_end_ndcg_minus_item_cf_ci95_high=intervals["end_to_end_minus_item_cf"][1],
        average_unique_candidates=float(np.mean(candidate_counts)),
    )


def _top_k_metrics(target: int, ranking: list[int], k: int) -> tuple[int, float]:
    """Return HitRate and single-positive NDCG for one already-ranked candidate list."""
    try:
        rank = ranking[:k].index(target) + 1
    except ValueError:
        return 0, 0.0
    return 1, 1.0 / math.log2(rank + 1)


def _ranker_features(
    candidate_indices: list[int],
    source_ranks: dict[int, tuple[int | None, int | None, int | None]],
    item_ids: list[str],
    history: list[int],
    history_length: int,
    metadata: dict[str, tuple[str, str]],
    popularity: TimeDecayedPopularity,
) -> np.ndarray:
    """Construct the exact numeric ranker feature contract used for cross-fitted training."""
    history_metadata = [
        metadata.get(item_ids[item_index], ("", "")) for item_index in history[-20:]
    ]
    rows: list[list[float]] = []
    for item_index in candidate_indices:
        item_id = item_ids[item_index]
        category, brand = metadata.get(item_id, ("", ""))
        neural_rank, popularity_rank, item_cf_rank = source_ranks[item_index]
        rows.append(
            [
                _inverse_rank(neural_rank),
                _inverse_rank(popularity_rank),
                _inverse_rank(item_cf_rank),
                float(sum(rank is not None for rank in source_ranks[item_index])),
                float(popularity.scores.get(item_id, 0.0)),
                float(history_length),
                _affinity(category, [value[0] for value in history_metadata]),
                _affinity(brand, [value[1] for value in history_metadata]),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _load_item_metadata(items_path: Path) -> dict[str, tuple[str, str]]:
    items = pl.read_parquet(items_path).select("item_id", "main_category", "brand")
    return {item_id: (category, brand) for item_id, category, brand in items.iter_rows()}


def _report(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)

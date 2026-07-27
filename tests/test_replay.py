from datetime import UTC, datetime, timedelta

import pytest

from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.data.replay import (
    derive_time_boundaries,
    is_replay_safe,
    iterative_k_core,
    split_name,
)
from rekindle.evaluation.bootstrap import bootstrap_mean_confidence_intervals
from rekindle.evaluation.replay import _ranker_features, _top_k_metrics
from rekindle.ranking.model import RANKER_FEATURES


def _timestamp(day: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day)


def test_global_split_is_chronological() -> None:
    boundaries = derive_time_boundaries(
        [_timestamp(day) for day in range(10)], validation_fraction=0.2, test_fraction=0.2
    )

    assert split_name(_timestamp(5), boundaries) == "train"
    assert split_name(_timestamp(6), boundaries) == "validation"
    assert split_name(_timestamp(8), boundaries) == "test"


def test_k_core_repeats_until_stable() -> None:
    interactions = [
        {"user_id": "u1", "item_id": "i1"},
        {"user_id": "u1", "item_id": "i2"},
        {"user_id": "u2", "item_id": "i2"},
    ]

    warm_users, warm_items = iterative_k_core(interactions, min_interactions=2)

    assert warm_users == set()
    assert warm_items == set()


def test_history_must_strictly_precede_target() -> None:
    target = _timestamp(3)

    assert is_replay_safe([{"event_ts": _timestamp(1)}, {"event_ts": _timestamp(2)}], target)
    assert not is_replay_safe([{"event_ts": target}], target)


def test_rejects_invalid_split_fractions() -> None:
    with pytest.raises(ValueError, match="less than one"):
        derive_time_boundaries([_timestamp(day) for day in range(4)], 0.6, 0.4)


def test_test_replay_builds_the_same_ranker_feature_shape_as_training() -> None:
    popularity = TimeDecayedPopularity(
        half_life_days=30,
        reference_time=_timestamp(1),
        scores={"a": 3.0, "c": 1.0},
    )

    features = _ranker_features(
        candidate_indices=[0, 2],
        source_ranks={0: (1, None, None), 2: (None, 2, 1)},
        item_ids=["a", "b", "c"],
        history=[1],
        history_length=7,
        metadata={"a": ("electronics", "brand-a"), "b": ("electronics", "brand-b")},
        popularity=popularity,
    )

    assert features.shape == (2, len(RANKER_FEATURES))
    assert features[0, 0] == 1.0
    assert features[1, 2] == 1.0
    assert features[0, 5] == 7.0


def test_top_k_metrics_scores_single_positive_by_its_rank() -> None:
    assert _top_k_metrics(target=5, ranking=[2, 5, 1], k=10) == (1, pytest.approx(1 / 1.5849625))
    assert _top_k_metrics(target=5, ranking=[2, 1, 5], k=2) == (0, 0.0)


def test_paired_bootstrap_intervals_are_reproducible_and_preserve_constant_scores() -> None:
    series = {
        "constant": [0.25, 0.25, 0.25],
        "paired_difference": [0.1, 0.0, -0.1],
    }

    first = bootstrap_mean_confidence_intervals(
        series, resamples=100, confidence_level=0.95, seed=42
    )
    second = bootstrap_mean_confidence_intervals(
        series, resamples=100, confidence_level=0.95, seed=42
    )

    assert first == second
    assert first["constant"] == (0.25, 0.25)

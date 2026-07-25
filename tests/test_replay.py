from datetime import UTC, datetime, timedelta

import pytest

from rekindle.data.replay import (
    derive_time_boundaries,
    is_replay_safe,
    iterative_k_core,
    split_name,
)


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

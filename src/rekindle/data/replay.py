"""Pure helpers for temporal splits, warm-set construction, and replay safety."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimeBoundaries:
    """The first timestamps assigned to validation and test windows."""

    validation_start: datetime
    test_start: datetime


def derive_time_boundaries(
    timestamps: Iterable[datetime], validation_fraction: float, test_fraction: float
) -> TimeBoundaries:
    """Create global chronological boundaries without looking at labels or users."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must each be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    ordered = sorted(timestamps)
    if len(ordered) < 3:
        raise ValueError("At least three timestamps are required for chronological splitting")

    validation_index = int(len(ordered) * (1 - validation_fraction - test_fraction))
    test_index = int(len(ordered) * (1 - test_fraction))
    if validation_index == test_index:
        raise ValueError("Not enough distinct positions for validation and test windows")

    return TimeBoundaries(
        validation_start=ordered[validation_index],
        test_start=ordered[test_index],
    )


def split_name(timestamp: datetime, boundaries: TimeBoundaries) -> str:
    """Assign one event to a global chronological split."""
    if timestamp < boundaries.validation_start:
        return "train"
    if timestamp < boundaries.test_start:
        return "validation"
    return "test"


def iterative_k_core(
    interactions: Iterable[Mapping[str, object]],
    min_interactions: int,
    user_key: str = "user_id",
    item_key: str = "item_id",
) -> tuple[set[str], set[str]]:
    """Return the bipartite user/item k-core using only supplied interactions.

    The caller controls the supplied interaction window. For Rekindle's warm set it must be
    training interactions only; filtering before the split would leak future activity.
    """
    if min_interactions < 1:
        raise ValueError("min_interactions must be positive")

    remaining = list(interactions)
    while True:
        user_counts = Counter(str(row[user_key]) for row in remaining)
        item_counts = Counter(str(row[item_key]) for row in remaining)
        next_remaining = [
            row
            for row in remaining
            if user_counts[str(row[user_key])] >= min_interactions
            and item_counts[str(row[item_key])] >= min_interactions
        ]
        if len(next_remaining) == len(remaining):
            return set(user_counts), set(item_counts)
        remaining = next_remaining


def is_replay_safe(history: Iterable[Mapping[str, object]], target_timestamp: datetime) -> bool:
    """Return whether every history event precedes the target event strictly."""
    return all(row["event_ts"] < target_timestamp for row in history)

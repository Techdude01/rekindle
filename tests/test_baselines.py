from datetime import UTC, datetime, timedelta

import polars as pl

from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity


def _event(item_id: str, days_before_cutoff: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "event_ts": datetime(2024, 1, 11, tzinfo=UTC) - timedelta(days=days_before_cutoff),
    }


def test_time_decayed_popularity_favors_recent_events_and_filters_seen_items() -> None:
    events = pl.DataFrame(
        [
            _event("old", 10),
            _event("new", 0),
        ]
    )

    model = TimeDecayedPopularity.fit(events, half_life_days=1)

    assert model.recommend(limit=2) == ["new", "old"]
    assert model.recommend(seen_item_ids={"new"}, limit=1) == ["old"]


def test_item_item_cosine_ranks_cooccurred_products_and_excludes_history() -> None:
    events = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["a", "b", "a", "b", "a", "c", "b", "c"],
            "event_ts": [datetime(2024, 1, 1, tzinfo=UTC)] * 8,
        }
    )

    model = ItemItemCosine.fit(events)

    assert model.recommend(history_item_ids=["a"], limit=2) == ["b", "c"]


def test_item_item_cosine_does_not_mutate_the_callers_seen_set() -> None:
    events = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["a", "b", "a", "c"],
            "event_ts": [datetime(2024, 1, 1, tzinfo=UTC)] * 4,
        }
    )
    seen = {"b"}

    ItemItemCosine.fit(events).recommend(history_item_ids=["a"], seen_item_ids=seen)

    assert seen == {"b"}

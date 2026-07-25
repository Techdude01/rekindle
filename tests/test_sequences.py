from datetime import UTC, datetime

import polars as pl

from rekindle.retrieval.sequences import build_sequence_examples


def _event(user_id: str, item_id: str, timestamp: str, split: str) -> dict[str, object]:
    return {
        "user_id": user_id,
        "item_id": item_id,
        "event_ts": datetime.fromisoformat(timestamp).replace(tzinfo=UTC),
        "split": split,
    }


def test_examples_require_two_prior_events_and_keep_same_timestamp_events_isolated() -> None:
    events = pl.DataFrame(
        [
            _event("u1", "a", "2024-01-01T00:00:00", "train"),
            _event("u1", "b", "2024-01-02T00:00:00", "train"),
            _event("u1", "c", "2024-01-03T00:00:00", "train"),
            _event("u1", "d", "2024-01-04T00:00:00", "train"),
            _event("u1", "e", "2024-01-05T00:00:00", "validation"),
            _event("u1", "f", "2024-01-05T00:00:00", "validation"),
            _event("u2", "e", "2024-01-01T00:00:00", "train"),
            _event("u2", "f", "2024-01-02T00:00:00", "train"),
        ]
    )

    train_examples = build_sequence_examples(
        events,
        target_split="train",
        history_size=4,
        min_prior_interactions=2,
    )
    validation_examples = build_sequence_examples(
        events,
        target_split="validation",
        history_size=4,
        min_prior_interactions=2,
    )

    item_index = {item_id: index for index, item_id in enumerate(train_examples.item_ids)}
    assert train_examples.count == 2
    assert train_examples.targets.tolist() == [item_index["c"], item_index["d"]]
    assert train_examples.histories.tolist() == [
        [-1, -1, item_index["a"], item_index["b"]],
        [-1, item_index["a"], item_index["b"], item_index["c"]],
    ]
    assert validation_examples.count == 2
    assert validation_examples.histories.tolist() == [
        [item_index["a"], item_index["b"], item_index["c"], item_index["d"]],
        [item_index["a"], item_index["b"], item_index["c"], item_index["d"]],
    ]

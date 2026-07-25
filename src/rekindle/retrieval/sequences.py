"""Build strict chronological next-item examples for retrieval training and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SequenceExamples:
    """Fixed-width recency histories and next-item targets from one chronological split."""

    user_indices: np.ndarray
    histories: np.ndarray
    targets: np.ndarray
    prior_event_counts: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    user_item_sequences: list[list[int]]

    @property
    def count(self) -> int:
        """Return the number of eligible next-item prediction examples."""
        return int(self.targets.shape[0])

    def save(self, directory: Path) -> None:
        """Persist local arrays and vocabulary mappings for reproducible training."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "examples.npz",
            user_indices=self.user_indices,
            histories=self.histories,
            targets=self.targets,
            prior_event_counts=self.prior_event_counts,
        )
        (directory / "mappings.json").write_text(
            json.dumps(
                {
                    "user_ids": self.user_ids,
                    "item_ids": self.item_ids,
                    "user_item_sequences": self.user_item_sequences,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> SequenceExamples:
        """Load locally persisted examples and mappings without touching raw review data."""
        arrays = np.load(directory / "examples.npz")
        mappings = json.loads((directory / "mappings.json").read_text(encoding="utf-8"))
        return cls(
            user_indices=arrays["user_indices"],
            histories=arrays["histories"],
            targets=arrays["targets"],
            prior_event_counts=arrays["prior_event_counts"],
            user_ids=mappings["user_ids"],
            item_ids=mappings["item_ids"],
            user_item_sequences=mappings["user_item_sequences"],
        )


def load_sequence_events(interactions_path: Path, target_split: str) -> pl.DataFrame:
    """Load warm train history plus the target split, never later interactions."""
    if target_split not in {"train", "validation", "test"}:
        raise ValueError("target_split must be train, validation, or test")

    allowed_splits = {
        "train": ["train"],
        "validation": ["train", "validation"],
        "test": ["train", "validation", "test"],
    }[target_split]
    return (
        pl.scan_parquet(interactions_path)
        .filter(
            pl.col("split").is_in(allowed_splits) & pl.col("is_warm_user") & pl.col("is_warm_item")
        )
        .select("user_id", "item_id", "event_ts", "split")
        .sort("user_id", "event_ts", "item_id")
        .collect()
    )


def build_sequence_examples(
    events: pl.DataFrame,
    target_split: str,
    history_size: int,
    min_prior_interactions: int,
) -> SequenceExamples:
    """Create examples using only events strictly earlier than each target timestamp.

    The warm vocabularies are derived from training events. Events tied at the same timestamp
    share identical prior histories, so none can leak into the others' feature context.
    """
    if history_size < 1:
        raise ValueError("history_size must be positive")
    if min_prior_interactions < 1:
        raise ValueError("min_prior_interactions must be positive")
    if events.is_empty():
        raise ValueError("Cannot build sequences from an empty event table")

    training_events = events.filter(pl.col("split") == "train")
    user_ids = sorted(training_events.get_column("user_id").unique().to_list())
    item_ids = sorted(training_events.get_column("item_id").unique().to_list())
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    user_indices: list[int] = []
    histories: list[list[int]] = []
    targets: list[int] = []
    prior_event_counts: list[int] = []
    user_item_sequences: list[list[int]] = [[] for _ in user_ids]

    active_user: str | None = None
    active_timestamp = None
    history: list[int] = []
    timestamp_events: list[tuple[str, str]] = []

    def flush_timestamp_events() -> None:
        if not timestamp_events:
            return
        user_index = user_to_index[active_user]  # type: ignore[index]
        if len(history) >= min_prior_interactions:
            padded_history = [-1] * max(0, history_size - len(history)) + history[-history_size:]
            for item_id, split in timestamp_events:
                if split == target_split and item_id in item_to_index:
                    user_indices.append(user_index)
                    histories.append(padded_history)
                    targets.append(item_to_index[item_id])
                    prior_event_counts.append(len(history))
        for item_id, _ in timestamp_events:
            if item_id in item_to_index:
                history.append(item_to_index[item_id])
                user_item_sequences[user_index].append(item_to_index[item_id])

    for user_id, item_id, event_ts, split in events.iter_rows():
        if user_id not in user_to_index:
            continue
        if active_user is None or user_id != active_user:
            flush_timestamp_events()
            active_user = user_id
            active_timestamp = event_ts
            history = []
            timestamp_events = []
        elif event_ts != active_timestamp:
            flush_timestamp_events()
            active_timestamp = event_ts
            timestamp_events = []
        timestamp_events.append((item_id, split))
    flush_timestamp_events()

    return SequenceExamples(
        user_indices=np.asarray(user_indices, dtype=np.int32),
        histories=(
            np.asarray(histories, dtype=np.int32).reshape(-1, history_size)
            if histories
            else np.empty((0, history_size), dtype=np.int32)
        ),
        targets=np.asarray(targets, dtype=np.int32),
        prior_event_counts=np.asarray(prior_event_counts, dtype=np.int32),
        user_ids=user_ids,
        item_ids=item_ids,
        user_item_sequences=user_item_sequences,
    )

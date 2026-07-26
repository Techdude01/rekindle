"""Sparse item–item cosine collaborative filtering for an interpretable baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse


@dataclass
class ItemItemCosine:
    """Score candidates by their cosine co-occurrence with recent user history."""

    interactions: sparse.csr_matrix
    item_ids: list[str]
    item_to_index: dict[str, int]
    item_norms: np.ndarray
    recency_decay: float

    @classmethod
    def fit(cls, events: pl.DataFrame, recency_decay: float = 0.8) -> ItemItemCosine:
        """Build a training-only sparse user-item matrix without materializing item pairs."""
        if not 0 < recency_decay <= 1:
            raise ValueError("recency_decay must be in (0, 1]")
        if events.is_empty():
            raise ValueError("Cannot fit item-item CF on an empty event table")

        user_ids = sorted(events.get_column("user_id").unique().to_list())
        item_ids = sorted(events.get_column("item_id").unique().to_list())
        user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

        rows = np.fromiter(
            (user_to_index[user_id] for user_id in events.get_column("user_id")),
            dtype=np.int32,
            count=events.height,
        )
        columns = np.fromiter(
            (item_to_index[item_id] for item_id in events.get_column("item_id")),
            dtype=np.int32,
            count=events.height,
        )
        values = np.ones(events.height, dtype=np.float32)
        interactions = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        ).tocsr()
        interactions.sum_duplicates()
        item_norms = np.sqrt(np.asarray(interactions.power(2).sum(axis=0)).ravel())

        return cls(
            interactions=interactions,
            item_ids=item_ids,
            item_to_index=item_to_index,
            item_norms=item_norms,
            recency_decay=recency_decay,
        )

    def recommend(
        self,
        history_item_ids: list[str],
        seen_item_ids: set[str] | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return unseen candidates scored from a chronologically ordered item history."""
        if limit < 1:
            raise ValueError("limit must be positive")

        history_indices = [
            self.item_to_index[item_id]
            for item_id in history_item_ids
            if item_id in self.item_to_index
        ]
        if not history_indices:
            return []

        source_norms = self.item_norms[history_indices]
        usable = source_norms > 0
        if not usable.any():
            return []
        history_indices = np.asarray(history_indices, dtype=np.int32)[usable]
        source_norms = source_norms[usable]
        recency_weights = self.recency_decay ** np.arange(len(history_indices) - 1, -1, -1)

        cooccurrence = self.interactions[:, history_indices].T @ self.interactions
        similarities = cooccurrence.tocoo(copy=True)
        denominator = source_norms[similarities.row] * self.item_norms[similarities.col]
        valid = denominator > 0
        similarities.data[valid] /= denominator[valid]
        similarities.data[~valid] = 0
        similarities.data *= recency_weights[similarities.row]
        scores = np.bincount(
            similarities.col,
            weights=similarities.data,
            minlength=len(self.item_ids),
        )

        seen = set(seen_item_ids or ())
        seen.update(history_item_ids)
        seen_indices = [
            self.item_to_index[item_id] for item_id in seen if item_id in self.item_to_index
        ]
        scores[seen_indices] = -np.inf
        available = np.flatnonzero(np.isfinite(scores))
        ranked = sorted(available, key=lambda index: (-scores[index], self.item_ids[index]))[:limit]
        return [self.item_ids[index] for index in ranked]

    def save(self, directory: Path) -> None:
        """Persist sparse matrix and item lookup table as local evaluation artifacts."""
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "interactions.npz", self.interactions)
        (directory / "items.json").write_text(
            json.dumps(
                {"item_ids": self.item_ids, "recency_decay": self.recency_decay},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> ItemItemCosine:
        """Load a persisted training-only sparse item-item model."""
        payload = json.loads((directory / "items.json").read_text(encoding="utf-8"))
        item_ids = payload["item_ids"]
        interactions = sparse.load_npz(directory / "interactions.npz").tocsr()
        item_norms = np.sqrt(np.asarray(interactions.power(2).sum(axis=0)).ravel())
        return cls(
            interactions=interactions,
            item_ids=item_ids,
            item_to_index={item_id: index for index, item_id in enumerate(item_ids)},
            item_norms=item_norms,
            recency_decay=float(payload["recency_decay"]),
        )

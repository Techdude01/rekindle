"""A time-decayed popularity baseline frozen at the end of the training window."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TimeDecayedPopularity:
    """Rank products by exponentially decayed interactions observed before a cutoff."""

    half_life_days: float
    reference_time: datetime
    scores: dict[str, float]

    @classmethod
    def fit(cls, events: pl.DataFrame, half_life_days: float) -> TimeDecayedPopularity:
        """Fit from training events only; ratings are not consumed."""
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if events.is_empty():
            raise ValueError("Cannot fit popularity on an empty event table")

        reference_time = max(events.get_column("event_ts"))
        scores: dict[str, float] = {}
        for item_id, event_ts in events.select("item_id", "event_ts").iter_rows():
            age_days = max((reference_time - event_ts).total_seconds() / 86_400, 0)
            contribution = 0.5 ** (age_days / half_life_days)
            scores[item_id] = scores.get(item_id, 0.0) + contribution

        return cls(
            half_life_days=half_life_days,
            reference_time=reference_time,
            scores=scores,
        )

    def recommend(self, seen_item_ids: set[str] | None = None, limit: int = 10) -> list[str]:
        """Return the highest-scoring unseen product IDs with deterministic tie breaking."""
        if limit < 1:
            raise ValueError("limit must be positive")
        seen = seen_item_ids or set()
        ranked = sorted(
            ((item_id, score) for item_id, score in self.scores.items() if item_id not in seen),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [item_id for item_id, _ in ranked[:limit]]

    def save(self, path: Path) -> None:
        """Persist a compact local artifact for later replay evaluation."""
        payload = {
            "half_life_days": self.half_life_days,
            "reference_time": self.reference_time.isoformat(),
            "scores": self.scores,
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TimeDecayedPopularity:
        """Load a persisted training-only popularity model."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            half_life_days=float(payload["half_life_days"]),
            reference_time=datetime.fromisoformat(payload["reference_time"]),
            scores={item_id: float(score) for item_id, score in payload["scores"].items()},
        )

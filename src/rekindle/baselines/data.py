"""Loading helpers that enforce the warm-corpus boundary for baseline models."""

from pathlib import Path

import polars as pl


def load_warm_events(interactions_path: Path, split: str) -> pl.DataFrame:
    """Load events whose user and item both belong to the training-only warm core."""
    return (
        pl.scan_parquet(interactions_path)
        .filter((pl.col("split") == split) & pl.col("is_warm_user") & pl.col("is_warm_item"))
        .select("user_id", "item_id", "event_ts")
        .sort("user_id", "event_ts", "item_id")
        .collect()
    )

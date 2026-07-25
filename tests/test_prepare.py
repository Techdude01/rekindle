import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from rekindle.config import load_config
from rekindle.data.prepare import prepare_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_preparation_builds_time_safe_canonical_tables(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    reviews = []
    metadata = []
    for user_number in range(5):
        for item_number in range(5):
            offset = user_number * 5 + item_number
            reviews.append(
                {
                    "user_id": f"user-{user_number}",
                    "parent_asin": f"item-{item_number}",
                    "timestamp": int((start + timedelta(days=offset)).timestamp() * 1000),
                    "rating": 1,
                    "verified_purchase": True,
                }
            )
    for item_number in range(5):
        metadata.append(
            {
                "parent_asin": f"item-{item_number}",
                "title": f"Item {item_number}",
                "main_category": "Electronics",
                "store": "Example brand",
            }
        )

    _write_jsonl(raw_directory / "reviews.jsonl", reviews)
    _write_jsonl(raw_directory / "metadata.jsonl", metadata)
    config = deepcopy(load_config(Path("config/base.yaml")))
    config["data"].update(
        {
            "raw_reviews_path": "raw/reviews.jsonl",
            "raw_metadata_path": "raw/metadata.jsonl",
            "prepared_interactions_path": "processed/interactions.parquet",
            "prepared_items_path": "processed/items.parquet",
            "min_train_interactions": 2,
        }
    )

    manifest_path = prepare_dataset(config, tmp_path)
    interactions = pl.read_parquet(tmp_path / "processed/interactions.parquet")
    items = pl.read_parquet(tmp_path / "processed/items.parquet")

    assert manifest_path.exists()
    assert interactions.height == 25
    assert {"user_id", "item_id", "event_ts", "split", "is_warm_user", "is_warm_item"} <= set(
        interactions.columns
    )
    assert "rating" not in interactions.columns
    assert items.height == 5
    assert interactions.filter(pl.col("split") == "train")["is_warm_user"].all()

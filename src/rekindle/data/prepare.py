"""Disk-backed preparation for Amazon Reviews 2023 JSONL files."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


def _quote_path(path: Path) -> str:
    """Quote a local path for a DuckDB SQL string literal."""
    return str(path.resolve()).replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary_from_epoch_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000, tz=UTC)


def prepare_dataset(config: dict[str, Any], project_root: Path) -> Path:
    """Build canonical Parquet tables and a data manifest without materializing raw JSONL.

    The warm user/item flags are calculated from training interactions only. Raw ratings are
    intentionally not selected into the canonical interaction table.
    """
    data_config = config["data"]
    split_config = config["split"]
    raw_reviews = project_root / data_config["raw_reviews_path"]
    raw_metadata = project_root / data_config["raw_metadata_path"]
    output_interactions = project_root / data_config["prepared_interactions_path"]
    output_items = project_root / data_config["prepared_items_path"]

    missing = [path for path in (raw_reviews, raw_metadata) if not path.exists()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing raw input file(s): {paths}")

    output_interactions.parent.mkdir(parents=True, exist_ok=True)
    output_items.parent.mkdir(parents=True, exist_ok=True)
    database_path = project_root / "data/interim/rekindle.duckdb"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = database_path.parent / "tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    review_path = _quote_path(raw_reviews)
    metadata_path = _quote_path(raw_metadata)
    interaction_path = _quote_path(output_interactions)
    item_path = _quote_path(output_items)

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("SET memory_limit = '8GB'")
        connection.execute(f"SET temp_directory = '{_quote_path(temp_directory)}'")
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE raw_events AS
            SELECT
                user_id::VARCHAR AS user_id,
                parent_asin::VARCHAR AS item_id,
                to_timestamp(timestamp / 1000.0) AS event_ts,
                COALESCE(verified_purchase, false)::BOOLEAN AS verified_purchase
            FROM read_json_auto('{review_path}')
            WHERE user_id IS NOT NULL
              AND parent_asin IS NOT NULL
              AND timestamp IS NOT NULL
            """
        )
        latest_timestamp = connection.execute("SELECT max(event_ts) FROM raw_events").fetchone()[0]
        window_start = connection.execute(
            "SELECT ? - ? * INTERVAL '1 month'",
            [latest_timestamp, data_config["recent_window_months"]],
        ).fetchone()[0]
        connection.execute(
            """
            CREATE OR REPLACE TABLE canonical_events AS
            SELECT user_id, item_id, event_ts, verified_purchase
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY user_id, item_id ORDER BY event_ts DESC
                    ) AS duplicate_rank
                FROM raw_events
                WHERE event_ts >= ?
            )
            WHERE duplicate_rank = 1
            """,
            [window_start],
        )

        epoch_boundaries = connection.execute(
            """
            SELECT
                quantile_cont(epoch_ms(event_ts), ?) AS validation_start_ms,
                quantile_cont(epoch_ms(event_ts), ?) AS test_start_ms
            FROM canonical_events
            """,
            [
                1 - split_config["validation_fraction"] - split_config["test_fraction"],
                1 - split_config["test_fraction"],
            ],
        ).fetchone()
        validation_start = _boundary_from_epoch_ms(epoch_boundaries[0])
        test_start = _boundary_from_epoch_ms(epoch_boundaries[1])
        if validation_start >= test_start:
            raise ValueError(
                "Temporal split requires a validation boundary before the test boundary"
            )
        connection.execute(
            """
            CREATE OR REPLACE TABLE split_events AS
            SELECT
                *,
                CASE
                    WHEN event_ts < ? THEN 'train'
                    WHEN event_ts < ? THEN 'validation'
                    ELSE 'test'
                END AS split
            FROM canonical_events
            """,
            [validation_start, test_start],
        )
        _apply_training_k_core(connection, data_config["min_train_interactions"])
        connection.execute(
            f"""
            COPY (
                SELECT
                    event.user_id,
                    event.item_id,
                    event.event_ts,
                    event.verified_purchase,
                    event.split,
                    warm_users.user_id IS NOT NULL AS is_warm_user,
                    warm_items.item_id IS NOT NULL AS is_warm_item
                FROM split_events AS event
                LEFT JOIN warm_users USING (user_id)
                LEFT JOIN warm_items USING (item_id)
            ) TO '{interaction_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    parent_asin::VARCHAR AS item_id,
                    COALESCE(title, '')::VARCHAR AS title,
                    COALESCE(main_category, '')::VARCHAR AS main_category,
                    COALESCE(store, '')::VARCHAR AS brand
                FROM read_json_auto('{metadata_path}')
                WHERE parent_asin IN (SELECT DISTINCT item_id FROM canonical_events)
            ) TO '{item_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = dict(
            zip(
                ["total_events", "train_events", "warm_users", "warm_items"],
                connection.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM split_events),
                        (SELECT count(*) FROM split_events WHERE split = 'train'),
                        (SELECT count(*) FROM warm_users),
                        (SELECT count(*) FROM warm_items)
                    """
                ).fetchone(),
                strict=True,
            )
        )

    manifest_path = output_interactions.parent / "data-manifest.json"
    manifest = {
        "source": data_config["source_name"],
        "raw_inputs": {
            str(raw_reviews): _sha256(raw_reviews),
            str(raw_metadata): _sha256(raw_metadata),
        },
        "recent_window_months": data_config["recent_window_months"],
        "min_train_interactions": data_config["min_train_interactions"],
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "counts": counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _apply_training_k_core(connection: duckdb.DuckDBPyConnection, minimum: int) -> None:
    """Materialize the iterative user/item k-core in DuckDB instead of in process memory."""
    connection.execute(
        "CREATE OR REPLACE TABLE core_events AS SELECT * FROM split_events WHERE split = 'train'"
    )
    while True:
        before_count = connection.execute("SELECT count(*) FROM core_events").fetchone()[0]
        connection.execute(
            """
            CREATE OR REPLACE TABLE next_core_events AS
            WITH user_counts AS (
                SELECT user_id, count(*) AS interaction_count FROM core_events GROUP BY user_id
            ), item_counts AS (
                SELECT item_id, count(*) AS interaction_count FROM core_events GROUP BY item_id
            )
            SELECT event.*
            FROM core_events AS event
            INNER JOIN user_counts USING (user_id)
            INNER JOIN item_counts USING (item_id)
            WHERE user_counts.interaction_count >= ?
              AND item_counts.interaction_count >= ?
            """,
            [minimum, minimum],
        )
        after_count = connection.execute("SELECT count(*) FROM next_core_events").fetchone()[0]
        connection.execute("DROP TABLE core_events")
        connection.execute("ALTER TABLE next_core_events RENAME TO core_events")
        if after_count == before_count:
            break

    connection.execute(
        "CREATE OR REPLACE TABLE warm_users AS SELECT DISTINCT user_id FROM core_events"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE warm_items AS SELECT DISTINCT item_id FROM core_events"
    )

"""Build leakage-resistant, cross-fitted candidate data for the LambdaRank stage."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from rekindle.baselines.item_cf import ItemItemCosine
from rekindle.baselines.popularity import TimeDecayedPopularity
from rekindle.evaluation.baselines import _first_unseen
from rekindle.evaluation.metrics import fixed_subset_indices
from rekindle.retrieval.inference import load_retriever, retrieve_top_k
from rekindle.retrieval.model import select_device
from rekindle.retrieval.sequences import SequenceExamples, build_sequence_examples
from rekindle.retrieval.training import train_retriever

ProgressCallback = Callable[[str], None]


def generate_cross_fitted_ranker_data(
    config: dict[str, Any], project_root: Path, progress_callback: ProgressCallback | None = None
) -> Path:
    """Create per-fold candidate Parquet only where a past-only retriever found the target."""
    interactions_path = project_root / config["data"]["prepared_interactions_path"]
    items_path = project_root / config["data"]["prepared_items_path"]
    output_directory = project_root / "artifacts/ranker-crossfit"
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = _load_item_metadata(items_path)
    fold_boundaries = _fold_boundaries(interactions_path, config["ranking"]["temporal_folds"])
    summaries: list[dict[str, int | float]] = []

    for fold_number, (start_ms, end_ms) in enumerate(fold_boundaries, start=1):
        _report(progress_callback, f"Fold {fold_number}: materializing its earlier-only 5-core.")
        training_events, target_events = _load_fold_events(
            interactions_path,
            start_ms,
            end_ms,
            config["data"]["min_train_interactions"],
            output_directory / f"fold-{fold_number}",
        )
        combined_events = pl.concat((training_events, target_events), how="vertical")
        train_examples = build_sequence_examples(
            training_events,
            target_split="train",
            history_size=config["split"]["replay_history_size"],
            min_prior_interactions=config["split"]["min_prior_interactions"],
        )
        target_examples = build_sequence_examples(
            combined_events,
            target_split="validation",
            history_size=config["split"]["replay_history_size"],
            min_prior_interactions=config["split"]["min_prior_interactions"],
        )
        query_indices = fixed_subset_indices(
            target_examples.count,
            config["ranking"]["queries_per_fold"],
            seed=config["project"]["seed"] + fold_number,
        )
        fold_directory = output_directory / f"fold-{fold_number}"
        _report(
            progress_callback,
            f"Fold {fold_number}: training a history-only retriever on "
            f"{train_examples.count:,} examples for {len(query_indices):,} later queries.",
        )
        train_retriever(
            train_examples,
            target_examples,
            config["retrieval"],
            seed=config["project"]["seed"] + fold_number,
            output_directory=fold_directory / "retriever",
            progress_callback=progress_callback,
            use_user_embedding=False,
            diagnostic_example_indices=query_indices,
        )
        device = select_device()
        retriever = load_retriever(fold_directory / "retriever/model.pt", device)
        neural_candidates = retrieve_top_k(
            retriever,
            target_examples,
            query_indices,
            k=config["retrieval"]["candidate_count"],
            batch_size=config["retrieval"]["evaluation_batch_size"],
            device=device,
            progress_callback=progress_callback,
        )
        popularity = TimeDecayedPopularity.fit(training_events, half_life_days=30)
        item_cf = ItemItemCosine.fit(
            training_events,
            recency_decay=config["baselines"]["item_cf_recency_decay"],
            history_size=config["baselines"]["item_cf_history_size"],
        )
        candidate_path = fold_directory / "candidates.parquet"
        summary = _write_ranker_candidates(
            candidate_path,
            fold_number,
            target_examples,
            query_indices,
            neural_candidates,
            popularity,
            item_cf,
            metadata,
            config["retrieval"]["candidate_count"],
            progress_callback,
        )
        summary["fold"] = fold_number
        summaries.append(summary)

    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps({"folds": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _fold_boundaries(interactions_path: Path, fold_count: int) -> list[tuple[float, float]]:
    """Split the global training window into rolling 25/25, 50/25, and 75/25 folds."""
    if fold_count != 3:
        raise ValueError("The ranker contract currently requires exactly three temporal folds")
    with duckdb.connect() as connection:
        boundaries = connection.execute(
            f"""
            SELECT quantile_cont(epoch_ms(event_ts), [0.25, 0.5, 0.75]),
                   max(epoch_ms(event_ts)) + 1
            FROM read_parquet('{_sql_path(interactions_path)}')
            WHERE split = 'train'
            """
        ).fetchone()
    quartiles = [float(value) for value in boundaries[0]]
    return list(zip(quartiles, [*quartiles[1:], float(boundaries[1])], strict=True))


def _load_fold_events(
    interactions_path: Path,
    start_ms: float,
    end_ms: float,
    minimum_core: int,
    fold_directory: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recompute a warm core from the period before a fold and load its next period."""
    fold_directory.mkdir(parents=True, exist_ok=True)
    database_path = fold_directory / "core.duckdb"
    temporary_directory = fold_directory / "tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("SET memory_limit = '8GB'")
        connection.execute(f"SET temp_directory = '{_sql_path(temporary_directory)}'")
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE core_events AS
            SELECT user_id, item_id, event_ts
            FROM read_parquet('{_sql_path(interactions_path)}')
            WHERE split = 'train' AND epoch_ms(event_ts) < ?
            """,
            [start_ms],
        )
        _apply_k_core(connection, minimum_core)
        connection.execute(
            "CREATE OR REPLACE TABLE warm_users AS SELECT DISTINCT user_id FROM core_events"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE warm_items AS SELECT DISTINCT item_id FROM core_events"
        )
        training = connection.execute(
            "SELECT user_id, item_id, event_ts, 'train' AS split FROM core_events "
            "ORDER BY user_id, event_ts, item_id"
        ).fetch_arrow_table()
        target = connection.execute(
            f"""
            SELECT event.user_id, event.item_id, event.event_ts, 'validation' AS split
            FROM read_parquet('{_sql_path(interactions_path)}') AS event
            INNER JOIN warm_users USING (user_id)
            INNER JOIN warm_items USING (item_id)
            WHERE event.split = 'train'
              AND epoch_ms(event.event_ts) >= ?
              AND epoch_ms(event.event_ts) < ?
            ORDER BY event.user_id, event.event_ts, event.item_id
            """,
            [start_ms, end_ms],
        ).fetch_arrow_table()
    return pl.from_arrow(training), pl.from_arrow(target)


def _apply_k_core(connection: duckdb.DuckDBPyConnection, minimum: int) -> None:
    """Iteratively retain only user/item pairs supported by earlier-fold interactions."""
    while True:
        before_count = connection.execute("SELECT count(*) FROM core_events").fetchone()[0]
        connection.execute(
            """
            CREATE OR REPLACE TABLE next_core_events AS
            WITH user_counts AS (
                SELECT user_id, count(*) AS event_count FROM core_events GROUP BY user_id
            ), item_counts AS (
                SELECT item_id, count(*) AS event_count FROM core_events GROUP BY item_id
            )
            SELECT event.*
            FROM core_events AS event
            INNER JOIN user_counts USING (user_id)
            INNER JOIN item_counts USING (item_id)
            WHERE user_counts.event_count >= ? AND item_counts.event_count >= ?
            """,
            [minimum, minimum],
        )
        after_count = connection.execute("SELECT count(*) FROM next_core_events").fetchone()[0]
        connection.execute("DROP TABLE core_events")
        connection.execute("ALTER TABLE next_core_events RENAME TO core_events")
        if after_count == before_count:
            return


def _load_item_metadata(items_path: Path) -> dict[str, tuple[str, str]]:
    """Load only category and brand fields used by the warm ranker features."""
    items = pl.read_parquet(items_path).select("item_id", "main_category", "brand")
    return {item_id: (category, brand) for item_id, category, brand in items.iter_rows()}


def _write_ranker_candidates(
    path: Path,
    fold_number: int,
    examples: SequenceExamples,
    query_indices: np.ndarray,
    neural_candidates: np.ndarray,
    popularity: TimeDecayedPopularity,
    item_cf: ItemItemCosine,
    metadata: dict[str, tuple[str, str]],
    candidate_count: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, int | float]:
    """Persist candidate rows only when the target was naturally retrieved by the union."""
    item_to_index = {item_id: index for index, item_id in enumerate(examples.item_ids)}
    popularity_ranked = [
        item_to_index[item_id]
        for item_id in sorted(
            popularity.scores, key=lambda item_id: (-popularity.scores[item_id], item_id)
        )
        if item_id in item_to_index
    ]
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, float | int | str]] = []
    retrieved_queries = 0
    total_rows = 0
    try:
        for query_number, example_index in enumerate(query_indices, start=1):
            user_index = int(examples.user_indices[example_index])
            position = int(examples.prior_event_counts[example_index])
            history = examples.user_item_sequences[user_index][:position]
            seen = set(history)
            target = int(examples.targets[example_index])
            neural = neural_candidates[query_number - 1].tolist()
            popular = _first_unseen(popularity_ranked, seen, candidate_count)
            collaborative = [
                item_cf.item_to_index[item_id]
                for item_id in item_cf.recommend(
                    history_item_ids=[examples.item_ids[item_index] for item_index in history],
                    seen_item_ids={examples.item_ids[item_index] for item_index in seen},
                    limit=candidate_count,
                )
            ]
            candidate_features = _merge_source_ranks(neural, popular, collaborative)
            if target not in candidate_features:
                continue
            retrieved_queries += 1
            query_id = fold_number * 10_000_000 + query_number
            history_metadata = [
                metadata.get(examples.item_ids[item_index], ("", ""))
                for item_index in history[-20:]
            ]
            for item_index, source_ranks in candidate_features.items():
                item_id = examples.item_ids[item_index]
                category, brand = metadata.get(item_id, ("", ""))
                rows.append(
                    {
                        "fold": fold_number,
                        "query_id": query_id,
                        "item_id": item_id,
                        "label": int(item_index == target),
                        "neural_inverse_rank": _inverse_rank(source_ranks[0]),
                        "popularity_inverse_rank": _inverse_rank(source_ranks[1]),
                        "item_cf_inverse_rank": _inverse_rank(source_ranks[2]),
                        "source_count": sum(rank is not None for rank in source_ranks),
                        "popularity_score": float(popularity.scores.get(item_id, 0.0)),
                        "user_history_length": position,
                        "category_affinity": _affinity(
                            category, [value[0] for value in history_metadata]
                        ),
                        "brand_affinity": _affinity(
                            brand, [value[1] for value in history_metadata]
                        ),
                    }
                )
            if len(rows) >= 100_000:
                writer = _write_rows(writer, path, rows)
                total_rows += len(rows)
                rows = []
            if progress_callback is not None and (
                query_number % 100 == 0 or query_number == len(query_indices)
            ):
                progress_callback(
                    f"Fold {fold_number} ranker candidates | query {query_number:,}/"
                    f"{len(query_indices):,}; retrieved targets {retrieved_queries:,}."
                )
        if rows:
            writer = _write_rows(writer, path, rows)
            total_rows += len(rows)
    finally:
        if writer is not None:
            writer.close()
    return {
        "sampled_queries": len(query_indices),
        "retrieved_queries": retrieved_queries,
        "candidate_rows": total_rows,
        "candidate_union_recall": retrieved_queries / len(query_indices),
    }


def _merge_source_ranks(
    neural: list[int], popularity: list[int], collaborative: list[int]
) -> dict[int, tuple[int | None, int | None, int | None]]:
    """Keep rank positions from every source without injecting a missing target."""
    candidates: dict[int, list[int | None]] = {}
    for source_index, source_candidates in enumerate((neural, popularity, collaborative)):
        for rank, item_index in enumerate(source_candidates, start=1):
            candidates.setdefault(item_index, [None, None, None])[source_index] = rank
    return {item_index: tuple(ranks) for item_index, ranks in candidates.items()}  # type: ignore[return-value]


def _inverse_rank(rank: int | None) -> float:
    """Represent source confidence by rank without comparing uncalibrated source scores."""
    return 0.0 if rank is None else 1.0 / rank


def _affinity(value: str, history_values: list[str]) -> float:
    """Calculate category or brand affinity only from items already in the query history."""
    if not value or not history_values:
        return 0.0
    return sum(value == history_value for history_value in history_values) / len(history_values)


def _write_rows(
    writer: pq.ParquetWriter | None, path: Path, rows: list[dict[str, float | int | str]]
) -> pq.ParquetWriter:
    """Append a bounded row batch to compressed candidate Parquet."""
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _sql_path(path: Path) -> str:
    """Quote a path for DuckDB SQL string literal use."""
    return str(path.resolve()).replace("'", "''")


def _report(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)

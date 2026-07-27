"""Fit and persist a LambdaRank reranker from cross-fitted candidate Parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

RANKER_FEATURES = (
    "neural_inverse_rank",
    "popularity_inverse_rank",
    "item_cf_inverse_rank",
    "source_count",
    "popularity_score",
    "user_history_length",
    "category_affinity",
    "brand_affinity",
)


@dataclass(frozen=True)
class RankerResult:
    """Selected LambdaRank complexity and its held-out cross-fitted quality."""

    best_iteration: int
    validation_ndcg_at_10: float
    training_queries: int
    validation_queries: int


def train_ranker(
    candidate_directory: Path,
    ranking_config: dict[str, Any],
    seed: int,
    output_directory: Path,
) -> RankerResult:
    """Select on fold 3, then refit the chosen LambdaRank capacity on every fold."""
    candidates = _load_candidates(candidate_directory)
    folds = sorted(candidates.get_column("fold").unique().to_list())
    if folds != [1, 2, 3]:
        raise ValueError("Ranker training requires exactly the three configured cross-fitted folds")
    training = candidates.filter(pl.col("fold") < 3)
    validation = candidates.filter(pl.col("fold") == 3)
    selection_model = _new_ranker(ranking_config, seed)
    selection_model.fit(
        _features(training),
        _labels(training),
        group=_group_sizes(training),
        eval_set=[(_features(validation), _labels(validation))],
        eval_group=[_group_sizes(validation)],
        eval_at=[10],
        callbacks=[
            lgb.early_stopping(ranking_config["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=25),
        ],
    )
    best_iteration = selection_model.best_iteration_ or ranking_config["n_estimators"]
    validation_ndcg = float(selection_model.best_score_["valid_0"]["ndcg@10"])

    final_model = _new_ranker(ranking_config, seed, n_estimators=best_iteration)
    final_model.fit(
        _features(candidates),
        _labels(candidates),
        group=_group_sizes(candidates),
        callbacks=[lgb.log_evaluation(period=25)],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(output_directory / "model.txt"))
    result = RankerResult(
        best_iteration=best_iteration,
        validation_ndcg_at_10=validation_ndcg,
        training_queries=len(_group_sizes(training)),
        validation_queries=len(_group_sizes(validation)),
    )
    (output_directory / "training-result.json").write_text(
        json.dumps(result.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "feature-importances.json").write_text(
        json.dumps(
            dict(zip(RANKER_FEATURES, final_model.feature_importances_.tolist(), strict=True)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _load_candidates(candidate_directory: Path) -> pl.DataFrame:
    """Read all non-empty fold Parquets in deterministic query order."""
    candidate_paths = sorted(candidate_directory.glob("fold-*/candidates.parquet"))
    if len(candidate_paths) != 3:
        raise FileNotFoundError("Expected candidates.parquet for each of the three folds")
    return pl.read_parquet(candidate_paths).sort("fold", "query_id")


def _features(candidates: pl.DataFrame) -> np.ndarray:
    """Return a compact numeric matrix in the stable model feature order."""
    return candidates.select(RANKER_FEATURES).cast(pl.Float32).to_numpy()


def _labels(candidates: pl.DataFrame) -> np.ndarray:
    """Return binary relevance labels for LambdaRank."""
    return candidates.get_column("label").to_numpy().astype(np.int32, copy=False)


def _group_sizes(candidates: pl.DataFrame) -> np.ndarray:
    """Return contiguous candidate counts for each query group."""
    return (
        candidates.group_by("query_id", maintain_order=True)
        .len()
        .get_column("len")
        .to_numpy()
        .astype(np.int32, copy=False)
    )


def _new_ranker(
    ranking_config: dict[str, Any], seed: int, n_estimators: int | None = None
) -> lgb.LGBMRanker:
    """Use a small CPU-friendly LambdaRank configuration with deterministic seeds."""
    return lgb.LGBMRanker(
        objective=ranking_config["objective"],
        metric="ndcg",
        n_estimators=n_estimators or ranking_config["n_estimators"],
        learning_rate=ranking_config["learning_rate"],
        num_leaves=ranking_config["num_leaves"],
        min_child_samples=ranking_config["min_child_samples"],
        reg_lambda=ranking_config["reg_lambda"],
        random_state=seed,
        # The full grouped matrix has crashed the macOS LightGBM runtime under
        # all-core native parallelism; one worker is slower but stable on 18 GB.
        n_jobs=ranking_config.get("n_jobs", 1),
        verbosity=-1,
    )

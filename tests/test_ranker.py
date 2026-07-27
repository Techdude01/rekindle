import polars as pl

from rekindle.ranking.model import RANKER_FEATURES, _group_sizes, _new_ranker


def test_ranker_group_sizes_preserve_contiguous_query_boundaries() -> None:
    candidates = pl.DataFrame(
        {
            "query_id": [10, 10, 11, 12, 12, 12],
            **{feature: [0.0] * 6 for feature in RANKER_FEATURES},
            "label": [1, 0, 1, 0, 1, 0],
        }
    )

    assert _group_sizes(candidates).tolist() == [2, 1, 3]


def test_ranker_uses_one_native_worker_on_the_reference_mac() -> None:
    ranker = _new_ranker(
        {
            "objective": "lambdarank",
            "n_estimators": 10,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
        },
        seed=42,
    )

    assert ranker.get_params()["n_jobs"] == 1

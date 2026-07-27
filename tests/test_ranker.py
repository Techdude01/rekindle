import polars as pl

from rekindle.ranking.model import RANKER_FEATURES, _group_sizes


def test_ranker_group_sizes_preserve_contiguous_query_boundaries() -> None:
    candidates = pl.DataFrame(
        {
            "query_id": [10, 10, 11, 12, 12, 12],
            **{feature: [0.0] * 6 for feature in RANKER_FEATURES},
            "label": [1, 0, 1, 0, 1, 0],
        }
    )

    assert _group_sizes(candidates).tolist() == [2, 1, 3]

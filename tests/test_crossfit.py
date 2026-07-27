from rekindle.ranking.crossfit import _merge_source_ranks


def test_cross_fitted_candidate_merge_keeps_only_source_candidates_and_their_ranks() -> None:
    candidates = _merge_source_ranks(
        neural=[7, 2],
        popularity=[2, 9],
        collaborative=[9, 4],
    )

    assert candidates == {
        7: (1, None, None),
        2: (2, 1, None),
        9: (None, 2, 1),
        4: (None, None, 2),
    }

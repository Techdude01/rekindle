"""Deterministic paired bootstrap intervals for offline replay metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_confidence_intervals(
    series: dict[str, np.ndarray],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return percentile intervals for paired per-event metrics.

    All input arrays must refer to the same replay events in the same order.  This
    lets callers bootstrap differences between systems without treating correlated
    event outcomes as independent samples.
    """
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence level must be between zero and one")
    names = list(series)
    values = np.column_stack([np.asarray(series[name], dtype=np.float64) for name in names])
    if values.shape[0] == 0:
        raise ValueError("Bootstrap requires at least one replay event")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap series must contain only finite values")

    random = np.random.default_rng(seed)
    estimates = np.empty((resamples, len(names)), dtype=np.float64)
    batch_size = 100
    for start in range(0, resamples, batch_size):
        batch_count = min(batch_size, resamples - start)
        indices = random.integers(0, values.shape[0], size=(batch_count, values.shape[0]))
        estimates[start : start + batch_count] = values[indices].mean(axis=1)

    tail = (1.0 - confidence_level) / 2.0
    bounds = np.quantile(estimates, [tail, 1.0 - tail], axis=0)
    return {
        name: (float(bounds[0, index]), float(bounds[1, index]))
        for index, name in enumerate(names)
    }

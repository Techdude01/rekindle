"""Small, deterministic utilities for comparable offline evaluation slices."""

import numpy as np


def fixed_subset_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    """Choose a reproducible subset without favoring early or late examples."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    random = np.random.default_rng(seed)
    return np.sort(random.choice(count, size=maximum, replace=False))

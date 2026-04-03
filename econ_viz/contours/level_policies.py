"""Shared contour-level policies."""

from __future__ import annotations

import numpy as np


def around_anchor_levels(anchor: float, n: int = 5, spread: float = 0.5) -> list[float]:
    """Compute n levels around an anchor utility value."""
    if n <= 1:
        return [anchor]

    lo = max(anchor * (1 - spread), 1e-6)
    hi = anchor * (1 + spread)
    n_below = n // 2
    n_above = n - n_below - 1
    below = np.linspace(lo, anchor, n_below + 1)[:-1].tolist()
    above = np.linspace(anchor, hi, n_above + 1)[1:].tolist()
    return below + [anchor] + above


def percentile_levels(Z: np.ndarray, n: int = 5, lo: float = 20, hi: float = 80) -> list[float]:
    """Compute contour levels from percentiles of a utility surface."""
    valid = Z[np.isfinite(Z)]
    return np.percentile(valid, np.linspace(lo, hi, n)).tolist()


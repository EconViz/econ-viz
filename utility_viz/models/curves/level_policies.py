"""Shared contour-level policies."""

from __future__ import annotations

import numpy as np


def around_anchor_levels(anchor: float, n: int = 5, spread: float = 0.5) -> list[float]:
    """Compute n levels around an anchor utility value."""
    if not np.isfinite(anchor):
        raise ValueError("anchor must be finite.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n <= 1:
        return [anchor]
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("spread must be finite and positive when n is greater than 1.")

    scale = abs(anchor) if anchor != 0 else 1.0
    half_width = scale * spread
    lo = anchor - half_width
    hi = anchor + half_width
    n_below = n // 2
    n_above = n - n_below - 1
    below = np.linspace(lo, anchor, n_below + 1)[:-1].tolist()
    above = np.linspace(anchor, hi, n_above + 1)[1:].tolist()
    return below + [anchor] + above


def percentile_levels(Z: np.ndarray, n: int = 5, lo: float = 20, hi: float = 80) -> list[float]:
    """Compute contour levels from percentiles of a utility surface."""
    valid = Z[np.isfinite(Z)]
    return np.percentile(valid, np.linspace(lo, hi, n)).tolist()

"""Backend-neutral level-curve geometry (numpy only, no plotting library)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Point = tuple[float, float]


def _bisect(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fixed: np.ndarray,
    lo: float,
    hi: float,
    level: float,
    *,
    vary_y: bool,
    iterations: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Root of ``func == level`` along one axis for every value of ``fixed``.

    Assumes *func* is non-decreasing along the varied axis. Returns the roots and a
    mask of the entries whose bracket ``[lo, hi]`` actually contains the level.
    """

    def evaluate(t: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(func(fixed, t) if vary_y else func(t, fixed), dtype=float)

    low = np.full_like(fixed, lo, dtype=float)
    high = np.full_like(fixed, hi, dtype=float)
    f_low, f_high = evaluate(low), evaluate(high)
    valid = np.isfinite(f_low) & np.isfinite(f_high) & (f_low <= level) & (level <= f_high)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        below = evaluate(mid) < level
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    return (low + high) / 2.0, valid


def level_curve_path(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    level: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    res: int = 400,
) -> list[Point]:
    """Trace the curve ``func(x, y) == level`` inside a rectangle as one polyline.

    The curve is located by bisection along columns and rows, so *func* must be
    non-decreasing in each good (true for the smooth, monotone preferences used
    by the Cobb-Douglas slice). Points are ordered by increasing ``x``. An empty
    list means the level never occurs inside the rectangle. Piecewise or
    non-monotone utilities need a dedicated tracer and are not handled here.
    """
    xs = np.linspace(x_range[0], x_range[1], res)
    ys = np.linspace(y_range[0], y_range[1], res)

    col_y, col_ok = _bisect(func, xs, y_range[0], y_range[1], level, vary_y=True)
    row_x, row_ok = _bisect(func, ys, x_range[0], x_range[1], level, vary_y=False)

    points = np.concatenate(
        [
            np.column_stack([xs[col_ok], col_y[col_ok]]),
            np.column_stack([row_x[row_ok], ys[row_ok]]),
        ]
    )
    if len(points) < 2:
        return []
    order = np.lexsort((-points[:, 1], points[:, 0]))
    ordered = points[order]
    keep = np.concatenate([[True], np.any(np.diff(ordered, axis=0) != 0, axis=1)])
    ordered = ordered[keep]
    if len(ordered) < 2:
        return []
    return [(float(x), float(y)) for x, y in ordered]

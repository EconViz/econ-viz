"""Backend-neutral level curves: bezierkit traces them, utility-viz chooses the levels."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from bezierkit.bezier.path import PiecewiseBezier
from bezierkit.implicit import ContourSet, trace_implicit
from bezierkit.sampling.uniform import UniformSampler

Point = tuple[float, float]

# Value substituted where a utility is undefined (e.g. Stone-Geary below subsistence), so
# the marching-squares grid stays finite. It is far below any real level, so a level
# crossing only ever lands next to the finite side of an undefined cell.
_UNDEFINED = -1e12


def trace_level_sets(
    func: Callable[..., Any],
    levels: Sequence[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    resolution: tuple[int, int] = (161, 161),
    tolerance: float = 0.005,
) -> ContourSet:
    """Trace ``func(x, y) == level`` for each level as cubic Bezier paths.

    Wraps :func:`bezierkit.implicit.trace_implicit`. Level *selection* stays with the
    caller (``percentile_levels`` and friends); this only converts levels to geometry.
    Handles non-monotone and kinked utilities (Leontief kinks are rounded to within
    *tolerance*; a level that is not present inside the box gives an empty path tuple).
    Disconnected components come back as separate paths.
    """

    def field(x: float, y: float) -> float:
        with np.errstate(all="ignore"):
            value = float(func(np.float64(x), np.float64(y)))
        return value if math.isfinite(value) else _UNDEFINED

    return trace_implicit(
        field,
        levels=list(levels),
        viewport=(x_range[0], x_range[1], y_range[0], y_range[1]),
        resolution=resolution,
        tolerance=tolerance,
    )


def sample_path(path: PiecewiseBezier, points_per_segment: int = 16) -> list[Point]:
    """Sample a Bezier path to an ordered polyline (what a raster backend draws)."""
    count = max(2, points_per_segment * len(path.segments))
    sample = UniformSampler(count).sample(path)
    return [(float(x), float(y)) for x, y in zip(sample.x, sample.y, strict=True)]

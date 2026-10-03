"""Curve post-processing used when drawing paths: spline smoothing and endpoint extension."""

from __future__ import annotations

import numpy as np

from utility_viz.core.constants.canvas import ENDPOINT_EXTENSION_FRAC, SMOOTH_SAMPLES


def _smooth_xy(xs: list[float], ys: list[float], n_samples: int = SMOOTH_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Return a parametric spline through ``(xs, ys)`` or the raw data as fallback."""
    if len(xs) < 3 or len(ys) < 3:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    try:
        from scipy.interpolate import make_interp_spline

        points = np.column_stack((xs, ys))
        diffs = np.diff(points, axis=0)
        chord = np.sqrt((diffs**2).sum(axis=1))
        t = np.concatenate(([0.0], np.cumsum(chord)))
        if np.isclose(t[-1], 0.0):
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        k = min(3, len(xs) - 1)
        t_new = np.linspace(0.0, t[-1], max(n_samples, len(xs)))
        spline_x = make_interp_spline(t, np.asarray(xs, dtype=float), k=k)
        spline_y = make_interp_spline(t, np.asarray(ys, dtype=float), k=k)
        return spline_x(t_new), spline_y(t_new)
    except Exception:  # pragma: no cover - fallback is intentionally conservative
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _extend_curve_endpoints(
    xs: np.ndarray,
    ys: np.ndarray,
    extension_frac: float = ENDPOINT_EXTENSION_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a curve slightly past its first and last points along endpoint tangents."""
    if len(xs) < 2 or len(ys) < 2 or extension_frac <= 0.0:
        return xs, ys

    start_vec = np.array([xs[1] - xs[0], ys[1] - ys[0]], dtype=float)
    end_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2]], dtype=float)
    total_scale = max(
        float(np.max(xs) - np.min(xs)),
        float(np.max(ys) - np.min(ys)),
        1.0,
    )

    def _extended_point(point_x: float, point_y: float, tangent: np.ndarray, sign: float) -> tuple[float, float]:
        norm = float(np.linalg.norm(tangent))
        if np.isclose(norm, 0.0):
            return point_x, point_y
        step = tangent / norm * (total_scale * extension_frac * sign)
        return point_x + step[0], point_y + step[1]

    start_x, start_y = _extended_point(xs[0], ys[0], start_vec, -1.0)
    end_x, end_y = _extended_point(xs[-1], ys[-1], end_vec, 1.0)
    return (
        np.concatenate(([start_x], xs, [end_x])),
        np.concatenate(([start_y], ys, [end_y])),
    )

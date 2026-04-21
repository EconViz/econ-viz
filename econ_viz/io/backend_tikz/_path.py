"""Geometry helpers for the TikZ renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def path_to_polylines(path: Path, transform, *, filled: bool) -> list[np.ndarray]:
    """Return transformed polylines for a Matplotlib path.

    ``Path.to_polygons`` is useful for filled regions and contour paths,
    but it returns an empty list for simple open line paths such as axis
    spines.  Fall back to ``iter_segments`` so those one-segment paths are
    still emitted.
    """
    polys = [strip_closing_vertex(poly, filled=filled) for poly in path.to_polygons(transform)]
    if polys:
        return polys
    return _segment_polylines(path, transform)


def _segment_polylines(path: Path, transform) -> list[np.ndarray]:
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for vertices, code in path.iter_segments(transform, simplify=False, curves=False):
        points = np.asarray(vertices, dtype=float).reshape(-1, 2)
        if code == Path.MOVETO:
            if current:
                polylines.append(current)
            current = [tuple(points[-1])]
        elif code == Path.CLOSEPOLY:
            if current:
                current.append(current[0])
                polylines.append(current)
                current = []
        else:
            if not current:
                current = [tuple(points[0])]
            current.append(tuple(points[-1]))

    if current:
        polylines.append(current)
    return [np.asarray(poly, dtype=float) for poly in polylines if len(poly) >= 2]


def strip_closing_vertex(poly: np.ndarray, *, filled: bool) -> np.ndarray:
    """Drop the auto-appended closing vertex for non-filled polygons.

    ``matplotlib.path.Path.to_polygons()`` returns closed sequences —
    the final vertex repeats the first so filled shapes can reuse the
    vertex list directly.  When the same vertices are stroked as an
    *open* curve (e.g. an indifference curve), the repetition draws a
    spurious diagonal segment back to the origin.  Strip it here so the
    downstream ``-- ``-joined TikZ coordinate list is clean.
    """
    if filled or len(poly) <= 2:
        return poly
    if np.allclose(poly[0], poly[-1]):
        return poly[:-1]
    return poly


def dash_spec(dashes) -> str:
    """Translate a Matplotlib dash array into a TikZ line-style option."""
    values = [float(d) for d in dashes]
    if len(values) == 2:
        on, off = values
        if on <= 2.0 and off >= on:
            return "dotted"
        if abs(on - off) <= max(on, off) * 0.25:
            return "dashed"
    parts = [
        f"{'on' if i % 2 == 0 else 'off'} {d:.2f}pt"
        for i, d in enumerate(values)
    ]
    return "dash pattern=" + " ".join(parts)

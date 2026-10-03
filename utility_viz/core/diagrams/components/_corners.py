"""Exact corner reconstruction for contours of kinked utilities.

``Axes.contour`` linearly interpolates inside every grid cell, so a cell that
contains a kink turns the right-angle corner into a short diagonal
("chamfer").  The functions here find such chamfers on the straight arms that
surround them and replace them with the single vertex where the two arms meet.
Curves without such a corner are returned untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path

# Edges whose directions differ by less than this (radians) form one straight run.
_COLLINEAR_TOL = 1e-6
# A corner needs the arms to turn by at least this much (radians).
_MIN_TURN = np.radians(20.0)
# Arms must be at least this many grid cells long, the chamfer at most this many.
_ARM_CELLS = 4.0
_CHAMFER_CELLS = 3.0
# A model kink point is adopted when within this many cell diagonals of the arm intersection.
_SNAP_CELLS = 2.0


def dedupe_consecutive(points: np.ndarray) -> np.ndarray:
    """Drop vertices identical to their predecessor."""
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points if keep.all() else points[keep]


def _runs(points: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs (first, last vertex) of maximal straight runs."""
    d = np.diff(points, axis=0)
    angle = np.arctan2(d[:, 1], d[:, 0])
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(d)):
        turn = abs((angle[i] - angle[i - 1] + np.pi) % (2 * np.pi) - np.pi)
        if turn > _COLLINEAR_TOL:
            runs.append((start, i))
            start = i
    runs.append((start, len(d)))
    return runs


def _intersect(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> np.ndarray | None:
    det = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(det) < 1e-12:
        return None
    t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / det
    return p0 + t * d0


def repair_corners(
    points: np.ndarray,
    cell: float,
    kinks: Sequence[tuple[float, float]] = (),
) -> np.ndarray:
    """Return *points* with chamfered corners replaced by exact vertices.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        An open contour polyline.
    cell : float
        Diagonal of one contour grid cell, in data units.
    kinks : sequence of (x, y)
        Exact kink points reported by the model; the nearest one is adopted
        when it lies close to the reconstructed corner.
    """
    points = dedupe_consecutive(points)
    if len(points) < 4:
        return points
    runs = _runs(points)
    if len(runs) < 3:
        return points

    def length(run: tuple[int, int]) -> float:
        return float(np.linalg.norm(points[run[1]] - points[run[0]]))

    def direction(run: tuple[int, int]) -> np.ndarray:
        return points[run[1]] - points[run[0]]

    out: list[np.ndarray] = []
    i = 0
    cursor = 0  # first vertex index not yet emitted
    changed = False
    while i < len(runs) - 2:
        if length(runs[i]) < _ARM_CELLS * cell:
            i += 1
            continue
        # Chain of short runs following the long arm i.
        j = i + 1
        chain = 0.0
        while j < len(runs) and length(runs[j]) < _ARM_CELLS * cell:
            chain += length(runs[j])
            j += 1
        if j == i + 1 or j >= len(runs) or chain > _CHAMFER_CELLS * cell:
            i = max(j, i + 1)
            continue
        d0, d1 = direction(runs[i]), direction(runs[j])
        cos = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1)))
        if np.arccos(np.clip(cos, -1.0, 1.0)) < _MIN_TURN:
            i = j
            continue
        found = _intersect(points[runs[i][0]], d0, points[runs[j][0]], d1)
        if found is None:
            i = j
            continue
        corner: np.ndarray = found
        if kinks:
            near = min(kinks, key=lambda k: float(np.hypot(k[0] - found[0], k[1] - found[1])))
            if np.hypot(near[0] - found[0], near[1] - found[1]) <= _SNAP_CELLS * cell:
                corner = np.array(near, dtype=float)
        # Vertices runs[i][1] .. runs[j][0] are the chamfer; the corner replaces them.
        out.append(points[cursor : runs[i][1]])
        out.append(corner[None, :])
        cursor = runs[j][0] + 1
        changed = True
        i = j
    if not changed:
        return points
    out.append(points[cursor:])
    return dedupe_consecutive(np.vstack(out))


def _repair_path(path: Path, cell: float, kinks: Sequence[tuple[float, float]]) -> Path:
    vertices: np.ndarray = np.asarray(path.vertices)
    if path.codes is None or len(vertices) < 2:
        return path
    codes: np.ndarray = np.asarray(path.codes)
    starts = np.flatnonzero(codes == Path.MOVETO)
    bounds = [*starts.tolist(), len(codes)]
    verts: list[np.ndarray] = []
    out_codes: list[np.ndarray] = []
    changed = False
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        sub = vertices[lo:hi]
        sub_codes = codes[lo:hi]
        if np.all(sub_codes[1:] == Path.LINETO):  # open polyline; closed loops are left alone
            fixed = repair_corners(sub, cell, kinks)
            if fixed is not sub:
                changed = True
                sub = fixed
                sub_codes = np.full(len(fixed), Path.LINETO, dtype=codes.dtype)
                sub_codes[0] = Path.MOVETO
        verts.append(sub)
        out_codes.append(sub_codes)
    if not changed:
        return path
    return Path(np.vstack(verts), np.concatenate(out_codes))


def repair_contour_set(cs, cell: float, kinks: Sequence[tuple[float, float]] = ()) -> None:
    """Replace chamfered corners in every path of contour set *cs* in place.

    Both Matplotlib rendering and the TikZ exporter read ``cs.get_paths()``,
    so updating the paths fixes every output format.  A contour set whose
    curves have no chamfered corner is left byte-for-byte unchanged.
    """
    paths = cs.get_paths()
    fixed = [_repair_path(p, cell, kinks) for p in paths]
    if any(new is not old for new, old in zip(fixed, paths, strict=True)):
        cs.set_paths(fixed)

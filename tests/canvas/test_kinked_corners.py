"""Kinked indifference curves must keep exact right-angle corners."""

import re

import numpy as np
import pytest

from utility_viz import Canvas
from utility_viz.core.diagrams.components import indifference
from utility_viz.models.utility import CobbDouglas, CustomUtility, Leontief


def _curve_polylines(canvas):
    """Contour polylines (data coordinates) of the first contour set."""
    cs = next(c for c in canvas.ax.collections if hasattr(c, "allsegs"))
    return [seg for level in cs.allsegs for seg in level if len(seg)]


def _leontief_canvas(level=6.0):
    c = Canvas()
    c.add_utility(Leontief(2, 3), levels=[level])
    return c


def test_leontief_vertices_lie_on_the_two_arms():
    (seg,) = _curve_polylines(_leontief_canvas())
    kx, ky = 6 / 2, 6 / 3
    on_vertical = np.isclose(seg[:, 0], kx, atol=1e-9) & (seg[:, 1] >= ky - 1e-9)
    on_horizontal = np.isclose(seg[:, 1], ky, atol=1e-9) & (seg[:, 0] >= kx - 1e-9)
    assert np.all(on_vertical | on_horizontal)


def test_leontief_corner_is_exact_kink_point_without_duplicates():
    (seg,) = _curve_polylines(_leontief_canvas())
    kx, ky = Leontief(2, 3).kink_points([6.0])[0]
    assert np.min(np.hypot(seg[:, 0] - kx, seg[:, 1] - ky)) < 1e-9
    assert np.all(np.any(np.diff(seg, axis=0) != 0, axis=1))


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, y: np.maximum(2 * x + y, x + 3 * y),
        lambda x, y: np.minimum(2 * x + y, x + 3 * y),
    ],
    ids=["max", "min"],
)
def test_piecewise_custom_utility_corner_is_exact(fn):
    level = 6.0
    c = Canvas()
    c.add_utility(CustomUtility(fn), levels=[level])
    (seg,) = _curve_polylines(c)
    # Kink locus 2x + y = x + 3y -> x = 2y; both pieces equal 5y there.
    ky = level / 5
    kx = 2 * ky
    assert np.min(np.hypot(seg[:, 0] - kx, seg[:, 1] - ky)) < 1e-9
    assert np.all(np.any(np.diff(seg, axis=0) != 0, axis=1))
    # Every vertex still lies on the level set (no off-curve chamfer points).
    assert np.allclose(fn(seg[:, 0], seg[:, 1]), level, atol=1e-9)


def test_smooth_cobb_douglas_segments_unchanged(monkeypatch):
    c = Canvas()
    c.add_utility(CobbDouglas(0.5, 0.5), levels=[1.0, 2.0])
    after = _curve_polylines(c)

    monkeypatch.setattr(indifference, "repair_contour_set", lambda *a, **k: None)
    c2 = Canvas()
    c2.add_utility(CobbDouglas(0.5, 0.5), levels=[1.0, 2.0])
    before = _curve_polylines(c2)

    assert len(before) == len(after) > 0
    for b, a in zip(before, after, strict=True):
        assert np.array_equal(b, a)


def test_tikz_export_has_exact_corner(tmp_path):
    out = tmp_path / "leontief.tex"
    _leontief_canvas().save(str(out))
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if "line width=1.80pt" in ln]
    assert len(lines) == 1
    pts = [tuple(map(float, m)) for m in re.findall(r"\(([-\d.]+),([-\d.]+)\)", lines[0])]
    assert len(pts) == 3
    (x0, _), (xc, yc), (_, y2) = pts
    assert xc == x0  # vertical arm ends exactly above the corner
    assert yc == y2  # horizontal arm starts exactly at the corner

"""add_decomposition draws the indifference curves through A, C, and (Slutsky) B."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, Label, Stroke
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)


def _dec(method):
    return decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=60.0, method=method)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _levels(ax):
    return sorted(float(level) for cs in _role(ax, "curve") for level in cs.levels)


def test_result_keeps_the_function():
    dec = _dec("hicks")
    assert dec.func is MODEL
    assert "func" not in repr(dec)


def test_hicks_draws_u0_and_u1():
    dec = _dec("hicks")
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(dec)
    assert _levels(cvs.ax) == pytest.approx(sorted([dec.A.utility, dec.C.utility]))
    assert dec.B.utility == pytest.approx(dec.A.utility)


def test_slutsky_adds_the_curve_through_b():
    dec = _dec("slutsky")
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(dec)
    assert _levels(cvs.ax) == pytest.approx(sorted([dec.A.utility, dec.B.utility, dec.C.utility]))


def test_show_curves_false_restores_previous_output():
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(_dec("slutsky"), show_curves=False)
    assert _role(cvs.ax, "curve") == []


def test_curves_use_the_ic_stroke_by_default():
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(_dec("hicks"))
    for cs in _role(cvs.ax, "curve"):
        assert to_hex(cs.get_edgecolor()[0]) == to_hex(cvs.theme.ic_color)


def test_curve_stroke():
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(
        _dec("hicks"), curve_stroke=Stroke(color="#123456", width=2.5, style="--"))
    for cs in _role(cvs.ax, "curve"):
        assert to_hex(cs.get_edgecolor()[0]) == "#123456"
        assert cs.get_linewidth()[0] == pytest.approx(2.5)


def test_no_curve_labels_by_default():
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(_dec("slutsky"))
    assert _role(cvs.ax, "ic_label") == []


def test_curve_labels():
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(
        _dec("slutsky"), curve_label=Label(position="top", color="#654321", text="ignored"))
    texts = _role(cvs.ax, "ic_label")
    assert sorted(t.get_text() for t in texts) == ["$U_0$", "$U_1$", "$U_B$"]
    assert all(to_hex(t.get_color()) == "#654321" for t in texts)


def test_decomposition_without_function_skips_curves():
    from dataclasses import replace

    cvs = Canvas(x_max=40, y_max=27).add_decomposition(replace(_dec("hicks"), func=None))
    assert _role(cvs.ax, "curve") == []


def test_range_arrows_point_from_start_to_end():
    dec = _dec("hicks")
    cvs = Canvas(x_max=40, y_max=27).add_decomposition(dec, show_x_projections=True)
    ranges = _role(cvs.ax, "range")
    assert len(ranges) == 2
    sub, inc = ranges
    # Substitution runs A -> B, income B -> C; the head sits at xy.
    assert (sub.xyann[0], sub.xy[0]) == pytest.approx((dec.A.x, dec.B.x))
    assert (inc.xyann[0], inc.xy[0]) == pytest.approx((dec.B.x, dec.C.x))
    assert all(type(r.arrow_patch.get_arrowstyle()).__name__ == "CurveB" for r in ranges)


def test_zero_effect_draws_no_range_arrow():
    import numpy as np

    from econ_viz.models import QuasiLinear

    # Quasi-linear in y: demand for x does not depend on income, so the income effect is zero.
    dec = decompose_price_effect(QuasiLinear(v_func=lambda z: 4.0 * np.log(z), linear_in="y"),
                                 px=(2.0, 1.0), py=1.0, income=28.0, method="hicks")
    assert dec.income_effect[0] == pytest.approx(0.0, abs=1e-3)
    cvs = Canvas(x_max=30, y_max=35).add_decomposition(dec, show_x_projections=True)
    ranges = _role(cvs.ax, "range")
    assert len(ranges) == 1
    assert (ranges[0].xyann[0], ranges[0].xy[0]) == pytest.approx((dec.A.x, dec.B.x))


def test_negative_utility_levels_are_solid():
    from econ_viz.models import Satiation

    cvs = Canvas(x_max=12, y_max=10).add_utility(Satiation(bliss_x=6, bliss_y=4), levels=[-8, -2])
    for cs in _role(cvs.ax, "curve"):
        assert all(style[1] is None for style in cs.get_linestyle())

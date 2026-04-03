"""Tests for the EdgeworthBox diagram helper."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pytest

from econ_viz import EdgeworthBox, EquilibriumFocusConfig
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, QuasiLinear


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_invalid_totals_raise():
    with pytest.raises(ValueError, match="positive"):
        EdgeworthBox(CobbDouglas(), CobbDouglas(), total_x=0.0, total_y=10.0)


def test_draw_indifference_curves_endowment_and_e_curves(tmp_path):
    box = EdgeworthBox(
        CobbDouglas(alpha=0.6, beta=0.4),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=8.0,
        title="Exchange economy",
    )
    box.add_indifference_curves(levels_a=4, levels_b=4).add_endowment(4.0, 3.0).add_endowment_indifference()
    out = tmp_path / "edgeworth_curves.png"
    box.save(str(out))
    assert out.exists()


def test_endowment_default_label_is_e():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.6, beta=0.4),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=8.0,
    )
    box.add_endowment(4.0, 3.0)
    labels = [text.get_text() for text in box.ax.texts]
    assert "$e$" in labels


def test_custom_utility_colors_can_be_set_in_constructor():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.6, beta=0.4),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=8.0,
        utility_a_color="#ff006e",
        utility_b_color="#3a86ff",
    )
    box.add_indifference_curves(levels_a=2, levels_b=2)
    colors = []
    for coll in box.ax.collections:
        edge = coll.get_edgecolor()
        if len(edge):
            colors.append(to_hex(edge[0]))
    assert "#ff006e" in colors
    assert "#3a86ff" in colors


def test_indifference_curves_can_be_generated_from_equilibrium():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    (
        box.add_endowment(4.0, 6.0)
        .add_contract_curve(n=80, method="mrs")
        .add_price_line(px=1.0, py=1.0)
        .add_walrasian_equilibrium(px=1.0, py=1.0)
        .add_indifference_curves_from_equilibrium(px=1.0, py=1.0, n_a=3, n_b=3)
    )
    assert box.walrasian_equilibrium is not None
    assert len(box.ax.collections) > 0


def test_equilibrium_indifference_draws_single_pair():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    (
        box.add_endowment(4.0, 6.0)
        .add_contract_curve(n=80, method="mrs")
        .add_price_line(px=1.0, py=1.0)
        .add_walrasian_equilibrium(px=1.0, py=1.0)
    )
    before = len(box.ax.collections)
    box.add_equilibrium_indifference(px=1.0, py=1.0)
    after = len(box.ax.collections)
    assert after >= before + 2


def test_equilibrium_focus_auto_can_skip_endowment_ic():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    (
        box.add_endowment(4.9, 5.1)
        .add_contract_curve(n=80, method="mrs")
        .add_price_line(px=1.0, py=1.0)
        .add_walrasian_equilibrium(px=1.0, py=1.0)
        .apply_equilibrium_focus(
            px=1.0,
            py=1.0,
            config=EquilibriumFocusConfig(
                include_endowment_indifference="auto",
                min_relative_gap=0.9,
            ),
        )
    )
    # Only equilibrium IC pair expected under high gap threshold.
    assert len(box.ax.collections) >= 2


def test_equilibrium_focus_renders_between_three_and_five_curves_per_agent():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.8, beta=0.2),
        CobbDouglas(alpha=0.2, beta=0.8),
        total_x=12.0,
        total_y=10.0,
    )
    (
        box.add_endowment(5.0, 4.0)
        .add_contract_curve(n=80, method="mrs")
        .add_price_line(px=1.2, py=1.0)
        .add_walrasian_equilibrium(px=1.2, py=1.0)
        .apply_equilibrium_focus(
            px=1.2,
            py=1.0,
            config=EquilibriumFocusConfig(
                include_endowment_indifference=True,
                min_curves_per_agent=3,
                max_curves_per_agent=5,
            ),
        )
    )
    assert 3 <= len(box.equilibrium_focus_levels_a) <= 5
    assert 3 <= len(box.equilibrium_focus_levels_b) <= 5


def test_contract_curve_default_style_is_dashed_and_thin():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=60, method="mrs")
    contract_lines = [line for line in box.ax.lines if line.get_label() == "Contract curve"]
    assert contract_lines
    line = contract_lines[0]
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() <= 1.2 + 1e-9


def test_symmetric_cobb_douglas_contract_curve_near_diagonal():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=70, method="mrs")
    pts = box.contract_curve_points
    assert len(pts) > 10
    assert np.median(np.abs(pts[:, 1] - pts[:, 0])) < 0.6


def test_asymmetric_cobb_douglas_contract_curve_is_skewed():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.8, beta=0.2),
        CobbDouglas(alpha=0.2, beta=0.8),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=80, method="mrs")
    pts = box.contract_curve_points
    assert len(pts) > 8
    assert float(np.median(pts[:, 1])) < 4.5


def test_perfect_substitutes_contract_curve_hits_boundaries():
    box = EdgeworthBox(
        PerfectSubstitutes(a=1.0, b=1.0),
        PerfectSubstitutes(a=1.0, b=1.0),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=50, method="pareto")
    pts = box.contract_curve_points
    assert len(pts) >= 2
    on_boundary = np.any(
        np.isclose(pts[:, 0], 0.0, atol=0.2)
        | np.isclose(pts[:, 0], 10.0, atol=0.2)
        | np.isclose(pts[:, 1], 0.0, atol=0.2)
        | np.isclose(pts[:, 1], 10.0, atol=0.2)
    )
    assert on_boundary


def test_leontief_contract_curve_is_computable():
    box = EdgeworthBox(
        Leontief(a=1.0, b=1.0),
        Leontief(a=1.0, b=1.0),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=60, method="pareto")
    pts = box.contract_curve_points
    assert len(pts) >= 2
    assert np.median(np.abs(pts[:, 1] - pts[:, 0])) < 1.0


def test_quasi_linear_contract_curve_concentrates_near_vertical():
    box = EdgeworthBox(
        QuasiLinear(v_func=np.log, linear_in="y"),
        QuasiLinear(v_func=np.log, linear_in="y"),
        total_x=10.0,
        total_y=10.0,
    )
    box.add_contract_curve(n=80, method="mrs")
    pts = box.contract_curve_points
    assert len(pts) >= 2
    assert abs(float(np.median(pts[:, 0])) - 5.0) < 0.7


def test_core_price_line_and_walrasian_equilibrium_checks():
    box = EdgeworthBox(
        CobbDouglas(alpha=0.5, beta=0.5),
        CobbDouglas(alpha=0.5, beta=0.5),
        total_x=10.0,
        total_y=10.0,
    )
    (
        box.add_endowment(4.0, 6.0)
        .add_endowment_indifference()
        .add_contract_curve(n=80, method="mrs")
        .add_core()
        .add_price_line(px=1.0, py=1.0)
        .add_walrasian_equilibrium(px=1.0, py=1.0)
    )
    assert len(box.core_points) > 0
    assert box.walrasian_equilibrium is not None
    x_star, y_star = box.walrasian_equilibrium
    checks = box.check_point(x_star, y_star, px=1.0, py=1.0)
    assert checks["market_clearing"]
    assert checks["budget_balance"]
    assert checks["individual_rationality"]


def test_endowment_outside_box_raises():
    box = EdgeworthBox(CobbDouglas(), CobbDouglas(), total_x=10.0, total_y=10.0)
    with pytest.raises(ValueError, match="inside"):
        box.add_endowment(11.0, 2.0)

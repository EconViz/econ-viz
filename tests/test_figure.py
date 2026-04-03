"""Tests for multi-panel figures, paths, and linked demand diagrams."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from econ_viz import Canvas, DemandDiagram, Figure, IncomePath, Layout, LinearBudget, PricePath
from econ_viz.models import CES, CobbDouglas, Leontief, PerfectSubstitutes, QuasiLinear, StoneGeary


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestFigure:
    @pytest.mark.parametrize("layout,expected", [
        (Layout.SINGLE, 1),
        (Layout.SIDE_BY_SIDE, 2),
        (Layout.TOP_TWO_BOTTOM_ONE, 3),
        (Layout.TOP_ONE_BOTTOM_TWO, 3),
        (Layout.GRID_2X2, 4),
        (Layout.GRID_3X3, 9),
    ])
    def test_all_required_layouts_create_expected_panel_count(self, layout, expected):
        fig = Figure(layout, x_max=12, y_max=10)
        assert len(fig) == expected

    def test_side_by_side_creates_two_canvases(self):
        fig = Figure(Layout.SIDE_BY_SIDE, x_max=12, y_max=10)
        assert len(fig) == 2
        assert isinstance(fig[0], Canvas)
        assert isinstance(fig[1], Canvas)

    def test_grid_lookup(self):
        fig = Figure(Layout.GRID_2X2)
        assert fig[1, 0] is fig[2]

    def test_save_png(self, tmp_path):
        out = tmp_path / "multi.png"
        fig = Figure(Layout.SIDE_BY_SIDE, x_max=10, y_max=10)
        fig[0].add_utility(CobbDouglas(), levels=3)
        fig[1].add_utility(CobbDouglas(alpha=0.3, beta=0.7), levels=3)
        fig.save(str(out))
        assert out.exists()

    def test_panel_canvas_uses_existing_api(self):
        fig = Figure(Layout.SIDE_BY_SIDE, x_max=20, y_max=15)
        fig[0].add_utility(CobbDouglas(), levels=3)
        fig[0].add_budget(px=2.0, py=3.0, income=30.0, fill=True)
        fig[1].add_point(2.0, 3.0, label="A")
        assert len(fig[0].ax.collections) >= 1
        assert len(fig[0].ax.lines) >= 1
        assert len(fig[1].ax.lines) >= 1

    def test_shared_axes_link_matplotlib_axes(self):
        fig = Figure(Layout.SIDE_BY_SIDE, shared_x=True, shared_y=True)
        assert fig[0].ax.get_shared_x_axes().joined(fig[0].ax, fig[1].ax)
        assert fig[0].ax.get_shared_y_axes().joined(fig[0].ax, fig[1].ax)

    def test_save_unsupported_format_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            Figure(Layout.SINGLE).save(str(tmp_path / "multi.tex"))


class TestConsumptionPaths:
    def test_price_path_collects_px_values(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = PricePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, price="px", price_range=(1.0, 4.0), n=4)
        assert path.parameter_name == "px"
        assert path.parameter_values == pytest.approx((1.0, 2.0, 3.0, 4.0))
        assert len(path.equilibria) == 4

    def test_price_path_can_vary_py(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = PricePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, price="py", price_range=(1.0, 4.0), n=4)
        assert path.parameter_name == "py"
        assert path.py_values == pytest.approx((1.0, 2.0, 3.0, 4.0))

    def test_income_path_collects_income_values(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = IncomePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, income_range=(10.0, 40.0), n=4)
        assert path.parameter_name == "income"
        assert path.parameter_values == pytest.approx((10.0, 20.0, 30.0, 40.0))
        assert len(path.equilibria) == 4

    def test_cobb_douglas_price_path_matches_closed_form(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = PricePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, price="px", price_range=(1.0, 4.0), n=4)
        for px, eq in zip(path.px_values, path.equilibria):
            assert eq.x == pytest.approx(20.0 / px, rel=1e-3)

    def test_price_path_defaults_to_single_smooth_curve_without_markers(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = PricePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, price="px", price_range=(1.0, 4.0), n=4)
        canvas = Canvas(x_max=25, y_max=25)
        before = len(canvas.ax.lines)
        canvas.add_path(path)
        assert len(canvas.ax.lines) == before + 1
        assert len(canvas.ax.lines[-1].get_xdata()) > len(path.equilibria)
        curve_xs = canvas.ax.lines[-1].get_xdata()
        assert curve_xs.min() < min(path.x_values)
        assert curve_xs.max() > max(path.x_values)

    def test_income_path_defaults_to_single_smooth_curve_without_markers(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = IncomePath(CobbDouglas(alpha=0.5, beta=0.5), budget=budget, income_range=(10.0, 40.0), n=4)
        canvas = Canvas(x_max=25, y_max=25)
        before = len(canvas.ax.lines)
        canvas.add_path(path)
        assert len(canvas.ax.lines) == before + 1
        assert len(canvas.ax.lines[-1].get_xdata()) > len(path.equilibria)
        curve_xs = canvas.ax.lines[-1].get_xdata()
        assert curve_xs.min() < min(path.x_values)
        assert curve_xs.max() > max(path.x_values)


class TestDemandDiagram:
    @pytest.mark.parametrize("model,budget,price,price_range", [
        (CobbDouglas(alpha=0.5, beta=0.5), LinearBudget(px=2.0, py=2.0, income=40.0), "px", (1.0, 4.0)),
        (CES(alpha=0.5, beta=0.5, rho=0.25), LinearBudget(px=2.0, py=2.0, income=40.0), "px", (1.0, 4.0)),
        (Leontief(a=1.0, b=1.0), LinearBudget(px=2.0, py=2.0, income=40.0), "px", (1.0, 4.0)),
        (PerfectSubstitutes(a=1.0, b=2.0), LinearBudget(px=2.0, py=2.0, income=40.0), "py", (1.0, 4.0)),
        (StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0), LinearBudget(px=2.0, py=2.0, income=20.0), "px", (1.0, 4.0)),
    ])
    def test_save_png_common_models(self, tmp_path, model, budget, price, price_range):
        out = tmp_path / f"demand_{price}.png"
        path = PricePath(
            model,
            budget=budget,
            price=price,
            price_range=price_range,
            n=20,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[1.5, 3.0]).save(str(out))
        assert out.exists()

    def test_selected_markers_lie_on_curve(self):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[2.0, 4.0])
        expected_x = [10.0, 5.0]
        for px, x in zip([2.0, 4.0], expected_x):
            assert x == pytest.approx(20.0 / px, rel=1e-3)

    @pytest.mark.parametrize(
        "price,expected",
        [
            ("px", r"p_{x}"),
            ("py", r"p_{y}"),
        ],
    )
    def test_demand_canvas_uses_math_price_symbols(self, price, expected):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price=price,
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        assert fig.demand_canvas.y_label == expected

    def test_quasi_linear_demand_example_renders(self, tmp_path):
        path = PricePath(
            QuasiLinear(v_func=np.log, linear_in="y"),
            budget=LinearBudget(px=2.0, py=1.0, income=2.0),
            price="px",
            price_range=(0.4, 5.0),
            n=20,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[0.8, 1.8])
        fig.save(str(tmp_path / "ql.png"))
        assert (tmp_path / "ql.png").exists()

    def test_demand_panel_draws_axis_guides_for_markers(self):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        before = len(fig.demand_canvas.ax.lines)
        fig.add_marshallian_panel(price_markers=[2.0, 4.0], show_demand_guides=True)
        after = len(fig.demand_canvas.ax.lines)
        assert after >= before + 6

    def test_perfect_substitutes_demand_breaks_at_tie(self):
        path = PricePath(
            PerfectSubstitutes(a=1.0, b=1.0),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(0.8, 5.0),
            n=21,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[1.2, 3.0])
        horizontal_segments = [
            line for line in fig.demand_canvas.ax.lines
            if len(line.get_xdata()) == 2 and len(set(line.get_ydata())) == 1
        ]
        assert horizontal_segments
        tie_markers = [
            line for line in fig.demand_canvas.ax.lines
            if len(line.get_xdata()) == 1
            and np.isclose(line.get_xdata()[0], 20.0)
            and np.isclose(line.get_ydata()[0], 2.0)
        ]
        assert tie_markers
        assert tie_markers[0].get_marker() == "o"
        assert tie_markers[0].get_color() == fig.demand_canvas.theme.ic_color
        assert tie_markers[0].get_clip_on() is False
        assert tie_markers[0].get_zorder() > horizontal_segments[0].get_zorder()

    def test_demand_markers_are_not_clipped_and_draw_above_lines(self):
        path = PricePath(
            PerfectSubstitutes(a=1.0, b=1.0),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(0.8, 5.0),
            n=21,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[3.0])
        point_markers = [
            line for line in fig.demand_canvas.ax.lines
            if len(line.get_xdata()) == 1
            and line.get_marker() == "o"
            and np.isclose(line.get_xdata()[0], 0.0)
            and np.isclose(line.get_ydata()[0], 3.0)
        ]
        assert point_markers
        assert point_markers[0].get_clip_on() is False
        curve_lines = [
            line for line in fig.demand_canvas.ax.lines
            if len(line.get_xdata()) > 1
        ]
        assert curve_lines
        assert point_markers[0].get_zorder() > curve_lines[0].get_zorder()

    def test_marshallian_panel_can_hide_legends(self):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(price_markers=[2.0, 4.0], show_legend=False)
        assert fig.utility_canvas.ax.get_legend() is None
        assert fig.demand_canvas.ax.get_legend() is None

    def test_goods_canvas_uses_wider_padding(self):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        assert fig.utility_canvas.x_max == pytest.approx(40.0 * 1.18)
        assert fig.utility_canvas.y_max == pytest.approx(20.0 * 1.18)

    def test_show_pcc_uses_path_color_not_ic_color(self):
        path = PricePath(
            CobbDouglas(alpha=0.5, beta=0.5),
            budget=LinearBudget(px=2.0, py=2.0, income=40.0),
            price="px",
            price_range=(1.0, 4.0),
            n=20,
        )
        fig = DemandDiagram(path)
        fig.add_marshallian_panel(show_pcc=True)
        pcc_line = fig.utility_canvas.ax.lines[-1]
        assert pcc_line.get_color() == fig.utility_canvas.theme.path_color
        assert fig.utility_canvas.theme.path_color != fig.utility_canvas.theme.ic_color

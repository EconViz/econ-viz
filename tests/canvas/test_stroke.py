"""Tests for Stroke: per-line width, style, colour, and arrowheads."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch

from econ_viz import ArrowStyle, Canvas, Figure, Layout, LineStyle, Stroke, levels, solve
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas, Leontief
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
EQ = solve(MODEL, px=2.0, py=3.0, income=30.0)
DASHED = Stroke(width=2.5, style="dashed", color="#123456")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _assert_stroked(artists, stroke=DASHED):
    assert artists, "no artists with this role"
    for artist in artists:
        assert np.atleast_1d(artist.get_linewidth())[0] == pytest.approx(stroke.width)
        colour = artist.get_color() if hasattr(artist, "get_color") else artist.get_edgecolor()
        if not isinstance(colour, str) and np.ndim(colour) == 2:
            colour = colour[0]
        assert to_hex(colour) == stroke.color


def _arrowheads(ax, role):
    return [p for p in ax.patches if isinstance(p, FancyArrowPatch) and getattr(p, "_ev_arrow_for", None) == role]


class TestStrokeValue:
    def test_defaults_leave_everything_unchanged(self):
        stroke = Stroke()
        assert (stroke.width, stroke.style, stroke.color, stroke.arrow) == (None, None, None, None)

    def test_normalises_style_and_arrow(self):
        stroke = Stroke(style="dotted", arrow="->")
        assert stroke.style is LineStyle.DOTTED
        assert stroke.arrow is ArrowStyle.SIMPLE

    @pytest.mark.parametrize("kwargs", [{"width": 0}, {"width": -1}, {"style": "wavy"}, {"arrow": "<<"}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Stroke(**kwargs)


class TestAxisStroke:
    def test_axis_stroke_applies_to_both_axes(self):
        cvs = Canvas(axis_stroke=Stroke(width=1.6, style="dashed", color="#aa0000", arrow="wedge"))
        for spine in ("bottom", "left"):
            assert cvs.ax.spines[spine].get_linewidth() == pytest.approx(1.6)
            assert cvs.ax.spines[spine].get_linestyle() == "dashed"
            assert to_hex(cvs.ax.spines[spine].get_edgecolor()) == "#aa0000"
        assert cvs.x_arrow_style is ArrowStyle.WEDGE
        assert cvs.y_arrow_style is ArrowStyle.WEDGE

    def test_per_axis_stroke_overrides_shared_stroke(self):
        cvs = Canvas(axis_stroke=Stroke(width=1.6), y_axis_stroke=Stroke(width=0.5, arrow="->"))
        assert cvs.ax.spines["bottom"].get_linewidth() == pytest.approx(1.6)
        assert cvs.ax.spines["left"].get_linewidth() == pytest.approx(0.5)
        assert cvs.x_arrow_style is ArrowStyle.TRIANGLE
        assert cvs.y_arrow_style is ArrowStyle.SIMPLE

    def test_figure_forwards_axis_strokes(self):
        fig = Figure(Layout.SIDE_BY_SIDE, x_axis_stroke=Stroke(width=2.0))
        assert all(p.ax.spines["bottom"].get_linewidth() == pytest.approx(2.0) for p in fig.canvases)


class TestLayerStrokes:
    def test_utility_curves_and_legend(self):
        cvs = Canvas().add_utility(MODEL, levels=[4, 5], label="IC", stroke=DASHED)
        _assert_stroked(_role(cvs.ax, "curve"))
        handle = next(h for h in cvs._legend_handles if h.get_label() == "IC")
        assert handle.get_linewidth() == pytest.approx(2.5)
        assert handle.get_linestyle() == "--"

    def test_utility_rays_have_their_own_stroke(self):
        cvs = Canvas().add_utility(Leontief(a=1, b=2), levels=[2, 4], show_rays=True, ray_stroke=DASHED)
        _assert_stroked(_role(cvs.ax, "ray"))

    def test_budget(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=True, stroke=DASHED)
        _assert_stroked(_role(cvs.ax, "budget"))

    def test_stroke_overrides_plain_arguments(self):
        cvs = Canvas().add_budget(2, 3, 30, linewidth=0.5, color="red", stroke=Stroke(width=3.0))
        line = _role(cvs.ax, "budget")[0]
        assert line.get_linewidth() == pytest.approx(3.0)
        assert to_hex(line.get_color()) == to_hex("red")

    def test_equilibrium_drop_lines_and_ray(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, show_ray=True, drop_stroke=DASHED, ray_stroke=Stroke(width=1.1))
        _assert_stroked(_role(cvs.ax, "drop"))
        assert _role(cvs.ax, "ray")[0].get_linewidth() == pytest.approx(1.1)

    def test_ray(self):
        cvs = Canvas().add_ray(0.5, stroke=DASHED)
        _assert_stroked(_role(cvs.ax, "ray"))

    def test_decomposition_budgets_and_effect_arrows(self):
        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        roles = ("original_budget", "compensated_budget", "final_budget", "substitution", "income")
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(dec, **{f"{r}_stroke": DASHED for r in roles})
        for role in roles[:3]:
            _assert_stroked(_role(cvs.ax, role))
        for role in roles[3:]:
            _assert_stroked([t.arrow_patch for t in _role(cvs.ax, role)])
        effect_handles = [h for h in cvs._legend_handles if getattr(h, "_ev_role", None) in roles[3:]]
        assert len(effect_handles) == 2
        assert all(h.get_linewidth() == pytest.approx(2.5) for h in effect_handles)

    def test_decomposition_projection_mode(self):
        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            dec, show_x_projections=True, projection_stroke=DASHED, guide_stroke=DASHED, range_stroke=DASHED,
        )
        for role in ("projection", "guide"):
            _assert_stroked(_role(cvs.ax, role))
        _assert_stroked([t.arrow_patch for t in _role(cvs.ax, "range")])


class TestArrowheads:
    @pytest.mark.parametrize("style", list(ArrowStyle))
    def test_any_line_can_end_in_an_arrow(self, style):
        cvs = Canvas().add_budget(2, 3, 30, stroke=Stroke(arrow=style))
        heads = _arrowheads(cvs.ax, "budget")
        assert len(heads) == 1
        assert heads[0]._ev_arrow_style is style

    def test_each_indifference_curve_gets_an_arrow(self):
        cvs = Canvas().add_utility(MODEL, levels=[3, 4, 5], stroke=Stroke(arrow="-|>"))
        assert len(_arrowheads(cvs.ax, "curve")) == 3

    def test_no_arrow_by_default(self):
        cvs = Canvas().add_budget(2, 3, 30)
        assert not _arrowheads(cvs.ax, "budget")

    def test_arrowheads_export_to_tikz(self, tmp_path):
        path = tmp_path / "arrow.tex"
        Canvas().add_ray(0.5, stroke=Stroke(arrow="-|>")).save(str(path))
        assert r"\filldraw" in path.read_text()


class TestThemeDefaults:
    """Without a Stroke, every line is drawn exactly as its theme default says."""

    @staticmethod
    def _matches(artist, stroke):
        width = np.atleast_1d(artist.get_linewidth())[0]
        style = artist.get_linestyle()
        style = style[0] if isinstance(style, list) else style
        colour = artist.get_color() if hasattr(artist, "get_color") else artist.get_edgecolor()
        if not isinstance(colour, str) and np.ndim(colour) == 2:
            colour = colour[0]
        mpl_style = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}[stroke.style.value]
        assert width == pytest.approx(stroke.width)
        assert style in (stroke.style.value, mpl_style) or isinstance(style, tuple)
        assert to_hex(colour) == to_hex(stroke.color)

    def test_layers_follow_theme(self):
        t = Canvas().theme
        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        cvs = (
            Canvas(x_max=20, y_max=15)
            .add_utility(MODEL, levels=[4, 5])
            .add_budget(2, 3, 30)
            .add_ray(0.5)
            .add_equilibrium(EQ)
        )
        self._matches(_role(cvs.ax, "curve")[0], t.ic_stroke)
        self._matches(_role(cvs.ax, "budget")[0], t.budget_stroke)
        self._matches(_role(cvs.ax, "ray")[0], t.ray_stroke)
        self._matches(_role(cvs.ax, "drop")[0], t.drop_stroke.merged_over(Stroke(color=t.eq_color)))

        cvs = Canvas(x_max=20, y_max=15).add_decomposition(dec)
        self._matches(_role(cvs.ax, "original_budget")[0], t.budget_stroke)
        self._matches(_role(cvs.ax, "compensated_budget")[0], t.compensated_budget_stroke)
        self._matches(_role(cvs.ax, "final_budget")[0], t.final_budget_stroke)

    def test_axes_follow_theme(self):
        cvs = Canvas()
        t = cvs.theme
        spine = cvs.ax.spines["bottom"]
        assert spine.get_linewidth() == pytest.approx(t.axis_stroke.width)
        assert spine.get_linestyle() == t.axis_stroke.style.value
        assert to_hex(spine.get_edgecolor()) == to_hex(t.axis_color)
        assert cvs.x_arrow_style is t.axis_stroke.arrow

    def test_custom_theme_changes_axis_default(self):
        from dataclasses import replace
        from econ_viz import themes

        theme = replace(themes.default, axis_stroke=Stroke(width=1.4, style="dashed", arrow="->"))
        cvs = Canvas(theme=theme)
        assert cvs.ax.spines["left"].get_linewidth() == pytest.approx(1.4)
        assert cvs.y_line_style is LineStyle.DASHED
        assert cvs.y_arrow_style is ArrowStyle.SIMPLE


class TestOtherDiagrams:
    def test_demand_diagram_panels(self):
        from econ_viz import DemandDiagram, LinearBudget, PricePath

        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=20)
        fig = DemandDiagram(path).add_marshallian_panel(
            price_markers=[1.5, 4.0], budget_stroke=DASHED, curve_stroke=DASHED,
            demand_stroke=DASHED, guide_stroke=DASHED,
        )
        _assert_stroked(_role(fig.utility_canvas.ax, "budget"))
        _assert_stroked(_role(fig.utility_canvas.ax, "curve"))
        _assert_stroked(_role(fig.demand_canvas.ax, "demand"))
        _assert_stroked(_role(fig.demand_canvas.ax, "guide"))
        legend_lines = fig.demand_canvas.ax.get_legend().get_lines()
        assert all(line.get_linewidth() == pytest.approx(2.5) for line in legend_lines)

    def test_edgeworth_box(self):
        from econ_viz.consumer.edgeworth import EdgeworthBox

        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0, box_stroke=Stroke(width=2.0, style="dashed"))
        assert box.ax.spines["top"].get_linewidth() == pytest.approx(2.0)
        assert box.ax.spines["top"].get_linestyle() == "dashed"
        box.add_endowment(7.0, 3.0)
        box.add_indifference_curves(levels_a=[3, 5], levels_b=[3, 5], stroke_a=DASHED, stroke_b=DASHED)
        box.add_contract_curve(stroke=DASHED)
        box.add_price_line(px=1.0, py=1.0, stroke=DASHED)
        for role in ("curve_a", "curve_b", "contract", "price"):
            _assert_stroked(_role(box.ax, role))

    def test_edgeworth_frame_follows_theme(self):
        from econ_viz.consumer.edgeworth import EdgeworthBox

        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0)
        assert box.ax.spines["top"].get_linewidth() == pytest.approx(box.theme.box_stroke.width)

"""Tests for Axis: one axis's label, label position, and stroke."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Axis, Canvas, DemandDiagram, Figure, Layout, LinearBudget, PricePath, Stroke
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.enums import ArrowStyle, LabelPosition, LineStyle
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas

MODEL = CobbDouglas(alpha=0.5, beta=0.5)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _axis_label(cvs, axis):
    return next(a for a in cvs.ax.get_children() if getattr(a, "_ev_axis_label", None) == axis)


class TestAxisValue:
    def test_defaults(self):
        assert (Axis().label, Axis().label_position, Axis().stroke) == (None, None, None)

    def test_position_accepts_strings(self):
        assert Axis(label_position="bottom").label_position is LabelPosition.BOTTOM

    def test_rejects_unknown_position(self):
        with pytest.raises(InvalidParameterError):
            Axis(label_position="middle")

    def test_canvas_rejects_position_invalid_for_that_axis(self):
        with pytest.raises(ValueError):
            Canvas(x_axis=Axis(label_position="left"))


class TestCanvasAxis:
    def test_without_axis_is_unchanged(self):
        plain = Canvas(x_label="q", x_label_pos="bottom", x_axis_stroke=Stroke(width=2))
        same = Canvas(x_axis=Axis(label="q", label_position="bottom", stroke=Stroke(width=2)))
        assert (plain.x_label, plain.x_label_pos, plain.x_axis_stroke) == (
            same.x_label,
            same.x_label_pos,
            same.x_axis_stroke,
        )
        assert _axis_label(plain, "x").get_text() == _axis_label(same, "x").get_text()

    def test_one_axis_sets_label_position_and_stroke(self):
        cvs = Canvas(
            y_axis=Axis(label="x_2", label_position="right", stroke=Stroke(width=1.7, style="--", arrow="-|>"))
        )
        assert cvs.y_label == "x_2" and cvs.y_label_pos is LabelPosition.RIGHT
        assert (cvs.y_axis_stroke.width, cvs.y_axis_stroke.style, cvs.y_axis_stroke.arrow) == (
            1.7,
            LineStyle.DASHED,
            ArrowStyle("-|>"),
        )
        assert cvs.x_label == "X" and cvs.x_axis_stroke == Canvas().x_axis_stroke
        assert cvs.ax.spines["left"].get_linewidth() == pytest.approx(1.7)

    def test_axis_beats_shorthand(self):
        cvs = Canvas(x_label="a", x_label_pos="top", x_axis=Axis(label="b", label_position="bottom"))
        assert (cvs.x_label, cvs.x_label_pos) == ("b", LabelPosition.BOTTOM)

    def test_unset_axis_fields_keep_shorthand(self):
        cvs = Canvas(x_label="a", x_label_pos="top", x_axis=Axis(stroke=Stroke(width=3)))
        assert (cvs.x_label, cvs.x_label_pos) == ("a", LabelPosition.TOP)

    def test_stroke_precedence(self):
        cvs = Canvas(
            x_line_style="dotted",
            x_arrow_style="-|>",
            axis_stroke=Stroke(width=1.0, color="#111111", style="--"),
            x_axis_stroke=Stroke(width=2.0, color="#222222"),
            x_axis=Axis(stroke=Stroke(width=3.0)),
        )
        stroke = cvs.x_axis_stroke
        assert stroke.width == 3.0  # Axis.stroke
        assert stroke.color == "#222222"  # x_axis_stroke
        assert stroke.style is LineStyle.DASHED  # axis_stroke
        assert stroke.arrow is ArrowStyle("-|>")  # x_arrow_style
        assert to_hex(cvs.ax.spines["bottom"].get_edgecolor()) == "#222222"


class TestOtherDiagrams:
    def test_figure_forwards_axis(self):
        fig = Figure(Layout.SIDE_BY_SIDE, x_axis=Axis(label="q", stroke=Stroke(width=2.5)))
        for cvs in fig.canvases:
            assert cvs.x_label == "q" and cvs.x_axis_stroke.width == 2.5

    def test_demand_diagram(self):
        path = PricePath(
            MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px", price_range=(0.8, 6.0), n=10
        )
        fig = DemandDiagram(path, x_axis=Axis(label="x_1", stroke=Stroke(width=2.5)))
        assert fig.utility_canvas.x_label == "x_1"
        assert fig.demand_canvas.x_label == "x"
        assert fig.demand_canvas.x_axis_stroke.width == 2.5

    def test_edgeworth(self):
        box = EdgeworthBox(
            MODEL, MODEL, total_x=10.0, total_y=10.0, x_axis=Axis(label="f", stroke=Stroke(width=3.0, color="#123456"))
        )
        assert box.x_label == "f"
        assert box.ax.get_xlabel() == "$f_A$"
        for side in ("bottom", "top"):
            assert box.ax.spines[side].get_linewidth() == pytest.approx(3.0)
            assert to_hex(box.ax.spines[side].get_edgecolor()) == "#123456"
        assert box.ax.spines["left"].get_linewidth() == pytest.approx(box.box_stroke.width)

    def test_edgeworth_rejects_label_position(self):
        with pytest.raises(InvalidParameterError):
            EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0, y_axis=Axis(label_position="top"))


def test_axis_is_exported():
    import econ_viz

    assert econ_viz.Axis is Axis

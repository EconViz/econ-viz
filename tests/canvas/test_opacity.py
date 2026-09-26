"""Opacity on every style object: Stroke, Marker, Label, Legend, Effect, Fill (#123)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from econ_viz import Axis, Canvas, Effect, Fill, Label, Legend, Marker, Stroke, solve
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.enums import ArrowStyle
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
EQ = solve(MODEL, px=2.0, py=3.0, income=30.0)
DEC = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


class TestValues:
    @pytest.mark.parametrize("cls", [Stroke, Marker, Label, Legend, Effect, Fill])
    @pytest.mark.parametrize("value", [-0.1, 1.2])
    def test_rejects_out_of_range(self, cls, value):
        with pytest.raises(InvalidParameterError):
            cls(opacity=value)

    @pytest.mark.parametrize("cls", [Stroke, Marker, Label, Legend, Fill])
    def test_merged_over_keeps_opacity(self, cls):
        assert cls().merged_over(cls(opacity=0.3)).opacity == 0.3
        assert cls(opacity=0.7).merged_over(cls(opacity=0.3)).opacity == 0.7

    def test_fill_alpha_is_shorthand(self):
        assert Fill(alpha=0.4).opacity == 0.4
        assert Fill(opacity=0.4).alpha == 0.4
        assert Fill("grey", 0.3).opacity == 0.3
        with pytest.raises(InvalidParameterError):
            Fill(alpha=0.2, opacity=0.5)


class TestDefaultsOpaque:
    def test_nothing_is_faded_by_default(self):
        cvs = Canvas(x_max=20, y_max=15, title="T").add_budget(2, 3, 30).add_equilibrium(EQ)
        for artist in (*_role(cvs.ax, "budget"), *_role(cvs.ax, "equilibrium"), *_role(cvs.ax, "equilibrium_label")):
            assert artist.get_alpha() is None
        assert cvs.ax.spines["bottom"].get_alpha() is None


class TestApplied:
    def test_stroke_and_arrowhead(self):
        cvs = Canvas().add_budget(2, 3, 30, stroke=Stroke(opacity=0.4, arrow=ArrowStyle.SIMPLE))
        assert _role(cvs.ax, "budget")[0].get_alpha() == 0.4
        heads = [p for p in cvs.ax.patches if getattr(p, "_ev_arrow_for", None) == "budget"]
        assert heads and heads[0].get_alpha() == 0.4

    def test_curve_collection(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5], stroke=Stroke(opacity=0.5))
        assert _role(cvs.ax, "curve")[0].get_alpha() == 0.5

    def test_axis_stroke(self):
        cvs = Canvas(x_axis=Axis(stroke=Stroke(opacity=0.3)))
        assert cvs.ax.spines["bottom"].get_alpha() == 0.3
        arrow = next(p for p in cvs.ax.patches if getattr(p, "_ev_axis_arrow", None) == "x")
        assert arrow.get_alpha() == 0.3
        assert cvs.ax.spines["left"].get_alpha() is None

    def test_marker_and_label(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, marker=Marker(opacity=0.6), label=Label(opacity=0.5))
        assert _role(cvs.ax, "equilibrium")[0].get_alpha() == 0.6
        assert _role(cvs.ax, "equilibrium_label")[0].get_alpha() == 0.5

    def test_axis_origin_and_title_labels(self):
        cvs = Canvas(title=Label(text="T", opacity=0.4), origin_label=Label(opacity=0.3),
                     x_axis=Axis(label=Label(text="q", opacity=0.2)))
        assert cvs.ax.title.get_alpha() == 0.4
        assert _role(cvs.ax, "origin_label")[0].get_alpha() == 0.3
        x_label = next(a for a in cvs.ax.get_children() if getattr(a, "_ev_axis_label", None) == "x")
        assert x_label.get_alpha() == 0.2

    def test_fill(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=Fill(opacity=0.25))
        assert _role(cvs.ax, "budget_fill")[0].get_alpha() == 0.25

    def test_effect(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            DEC, income=Effect(opacity=0.4, label="IE"), substitution=Effect(label="SE"))
        assert _role(cvs.ax, "income")[0].arrow_patch.get_alpha() == 0.4
        assert _role(cvs.ax, "substitution")[0].arrow_patch.get_alpha() is None
        assert _role(cvs.ax, "income_label")[0].get_alpha() == 0.4
        handle = next(h for h in cvs._legend_handles if getattr(h, "_ev_role", None) == "income")
        assert handle.get_alpha() == 0.4

    def test_effect_range_arrows(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            DEC, show_x_projections=True, substitution=Effect(opacity=0.3))
        alphas = sorted((r.arrow_patch.get_alpha() or 1.0) for r in _role(cvs.ax, "range"))
        assert alphas == [0.3, 1.0]

    def test_label_opacity_beats_effect_opacity(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            DEC, income=Effect(opacity=0.4, label=Label(text="IE", opacity=0.9)))
        assert _role(cvs.ax, "income_label")[0].get_alpha() == 0.9

    def test_legend(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(opacity=0.5, frame=True))
        legend = cvs.ax.get_legend()
        assert legend.get_frame().get_alpha() == 0.5
        assert all(t.get_alpha() == 0.5 for t in legend.get_texts())

    def test_edgeworth(self):
        from econ_viz import Axis as A

        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0,
                           x_axis=A(stroke=Stroke(opacity=0.3), label=Label(text="f", opacity=0.6)))
        assert box.ax.spines["bottom"].get_alpha() == 0.3
        assert box.ax.xaxis.label.get_alpha() == 0.6

    def test_opacity_reaches_tikz(self, tmp_path):
        path = tmp_path / "o.tex"
        Canvas().add_budget(2, 3, 30, stroke=Stroke(opacity=0.4)).save(str(path))
        assert "opacity" in path.read_text()

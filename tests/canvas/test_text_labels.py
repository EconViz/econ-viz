"""Every text element takes a Label: axis labels, origin, titles, effect labels, Edgeworth text (#122)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Axis, Canvas, Effect, Figure, Label, Layout
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.enums import LabelPosition
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
DEC = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _axis_label(cvs, axis):
    return next(a for a in cvs.ax.get_children() if getattr(a, "_ev_axis_label", None) == axis)


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _texts(ax, text):
    return [t for t in ax.texts if t.get_text() == text]


class TestDefaultsUnchanged:
    def test_canvas(self):
        cvs = Canvas(title="T")
        t = cvs.theme
        for axis in ("x", "y"):
            label = _axis_label(cvs, axis)
            assert label.get_fontsize() == 14
            assert to_hex(label.get_color()) == to_hex(t.label_color)
        assert tuple(_axis_label(cvs, "x").xyann) == (8, 0)
        origin = _role(cvs.ax, "origin_label")[0]
        assert origin.get_text() == "$0$" and origin.get_fontsize() == 12
        from matplotlib.font_manager import FontProperties

        default_size = FontProperties(size=plt.rcParams["axes.titlesize"]).get_size_in_points()
        assert cvs.ax.title.get_fontsize() == pytest.approx(default_size)

    def test_effect_label_size(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, substitution=Effect(label="SE"))
        assert _role(cvs.ax, "substitution_label")[0].get_fontsize() == 10


class TestCanvasText:
    def test_axis_label_as_label(self):
        cvs = Canvas(x_axis=Axis(label=Label(text="x_1", fontsize=18, color="#123456", offset=16)))
        label = _axis_label(cvs, "x")
        assert label.get_text() == "$x_1$"
        assert label.get_fontsize() == 18
        assert to_hex(label.get_color()) == "#123456"
        assert tuple(label.xyann) == (16, 0)

    def test_axis_label_position_from_label(self):
        cvs = Canvas(y_axis=Axis(label=Label(text="q", position="right")))
        assert cvs.y_label_pos is LabelPosition.RIGHT
        cvs = Canvas(y_axis=Axis(label=Label(text="q", position="right"), label_position="left"))
        assert cvs.y_label_pos is LabelPosition.LEFT

    def test_axis_label_without_text_keeps_shorthand(self):
        cvs = Canvas(x_label="q", x_axis=Axis(label=Label(fontsize=9)))
        assert _axis_label(cvs, "x").get_text() == "$q$"
        assert _axis_label(cvs, "x").get_fontsize() == 9

    def test_hide_axis_label(self):
        cvs = Canvas(y_axis=Axis(label=Label(visible=False)))
        assert not _axis_label(cvs, "y").get_visible()

    def test_origin_label(self):
        cvs = Canvas(origin_label=Label(fontsize=8, color="#654321"))
        origin = _role(cvs.ax, "origin_label")[0]
        assert origin.get_fontsize() == 8 and to_hex(origin.get_color()) == "#654321"
        assert _role(Canvas(origin_label="O").ax, "origin_label")[0].get_text() == "$O$"
        assert not _role(Canvas(origin_label=Label(visible=False)).ax, "origin_label")[0].get_visible()

    def test_title_as_label(self):
        cvs = Canvas(title=Label(text="Hicks", fontsize=20, color="#112233"))
        assert cvs.ax.title.get_text() == "Hicks"
        assert cvs.ax.title.get_fontsize() == 20
        assert to_hex(cvs.ax.title.get_color()) == "#112233"

    def test_figure_title_as_label(self):
        fig = Figure(Layout.SIDE_BY_SIDE, title=Label(text="Panels", fontsize=17))
        assert fig.fig._suptitle.get_text() == "Panels"
        assert fig.fig._suptitle.get_fontsize() == 17

    def test_axis_label_reaches_tikz(self, tmp_path):
        path = tmp_path / "t.tex"
        Canvas(x_axis=Axis(label=Label(text="q_1", fontsize=18))).save(str(path))
        assert "q_1" in path.read_text()


class TestEffectLabel:
    def test_label_object(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            DEC, substitution=Effect(label=Label(text="SE", fontsize=13, color="#123456")))
        text = _role(cvs.ax, "substitution_label")[0]
        assert text.get_text() == "SE" and text.get_fontsize() == 13
        assert to_hex(text.get_color()) == "#123456"

    def test_label_position_beats_shorthand(self):
        effect = Effect(label=Label(text="SE", position="top", offset=9), label_position="bottom", label_offset=2)
        text, style = effect.resolved_label(Label(fontsize=10))
        assert (text, style.position, style.offset, style.fontsize) == ("SE", LabelPosition.TOP, 9, 10)

    def test_shorthand_still_works(self):
        text, style = Effect(label="IE", label_position="left", label_offset=7).resolved_label(Label())
        assert (text, style.position, style.offset) == ("IE", LabelPosition.LEFT, 7)

    def test_hidden(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(
            DEC, income=Effect(label=Label(text="IE", visible=False)))
        assert not _role(cvs.ax, "income_label")[0].get_visible()


class TestEdgeworthText:
    def _box(self, **kwargs):
        return EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0, **kwargs)

    def test_defaults_unchanged(self):
        box = self._box()
        assert box.ax.get_xlabel() == "$x_A$"
        assert _texts(box.ax, "$O_A$") and _texts(box.ax, "$x_B$")

    def test_good_names_as_label(self):
        box = self._box(x_axis=Axis(label=Label(text="f", fontsize=16, color="#123456")))
        assert box.ax.get_xlabel() == "$f_A$"
        assert box.ax.xaxis.label.get_fontsize() == 16
        other = _texts(box.ax, "$f_B$")[0]
        assert other.get_fontsize() == 16 and to_hex(other.get_color()) == "#123456"

    def test_origin_label(self):
        box = self._box(origin_label=Label(fontsize=15))
        assert all(t.get_fontsize() == 15 for t in _texts(box.ax, "$O_A$") + _texts(box.ax, "$O_B$"))
        box = self._box(origin_label=Label(visible=False))
        assert not _texts(box.ax, "$O_A$")[0].get_visible()

    def test_title_as_label(self):
        box = self._box(title=Label(text="Exchange", fontsize=19))
        assert box.ax.title.get_text() == "Exchange" and box.ax.title.get_fontsize() == 19


def test_theme_defaults():
    t = Canvas().theme
    assert (t.axis_label.fontsize, t.axis_label.offset) == (14, 8)
    assert (t.origin_label.text, t.origin_label.fontsize) == ("0", 12)
    assert t.effect_label.fontsize == 10
    assert t.title_label == Label() and t.box_label == Label()

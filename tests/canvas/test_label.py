"""Tests for Label: text, position, offset, colour, size, and visibility of point labels."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, DemandDiagram, Label, LinearBudget, Marker, PricePath, solve
from econ_viz.canvas.labels import placement
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.enums import LabelPosition
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas, Satiation
from econ_viz.optimizer import decompose_price_effect
from econ_viz.themes.label import split_label

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
EQ = solve(MODEL, px=2.0, py=3.0, income=30.0)
DEC = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _placed(text, position, offset):
    xytext, ha, va = placement(LabelPosition(position), offset)
    assert tuple(text.xyann) == pytest.approx(xytext)
    assert (text.get_horizontalalignment(), text.get_verticalalignment()) == (ha, va)


class TestLabelValue:
    def test_defaults(self):
        label = Label()
        assert (label.text, label.position, label.offset, label.color, label.fontsize, label.visible) == (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def test_position_accepts_strings_and_aliases(self):
        assert Label(position="top-left").position is LabelPosition.TOP_LEFT
        assert Label(position="above").position is LabelPosition.TOP
        assert Label(position="below").position is LabelPosition.BOTTOM

    @pytest.mark.parametrize("kwargs", [{"position": "middle"}, {"fontsize": 0}, {"fontsize": -1}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Label(**kwargs)

    def test_merged_over_keeps_unset_fields(self):
        merged = Label(offset=9).merged_over(Label(position="left", offset=2, color="red"))
        assert (merged.position, merged.offset, merged.color) == (LabelPosition.LEFT, 9, "red")

    def test_split_label(self):
        default = Label(position="top-right", offset=5)
        assert split_label("P", default) == ("P", default)
        assert split_label(None, default, "x^*") == (None, default)
        text, style = split_label(Label(position="left"), default, "x^*")
        assert text == "x^*"
        assert (style.text, style.position, style.offset) == (None, LabelPosition.LEFT, 5)

    @pytest.mark.parametrize("position", [p.value for p in LabelPosition])
    def test_placement_points_away_from_the_anchor(self, position):
        (dx, dy), ha, va = placement(LabelPosition(position), 4.0)
        assert (dx > 0) == (ha == "left") and (dx < 0) == (ha == "right")
        assert (dy > 0) == (va == "bottom") and (dy < 0) == (va == "top")


class TestDefaultsUnchanged:
    def test_point_and_equilibrium(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ).add_point(2, 2, label="P")
        for role in ("equilibrium_label", "point_label"):
            text = _role(cvs.ax, role)[0]
            assert tuple(text.xyann) == (5, 5)
            assert text.get_fontsize() == 12
        assert _role(cvs.ax, "equilibrium_label")[0].get_text() == "$x^*$"
        assert _role(cvs.ax, "point_label")[0].get_text() == "$P$"

    def test_equilibrium_label_none_draws_nothing(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, label=None)
        assert _role(cvs.ax, "equilibrium_label") == []

    def test_legacy_offset_still_works(self):
        cvs = Canvas().add_point(2, 2, label="P", offset=(-10, 3))
        assert tuple(_role(cvs.ax, "point_label")[0].xyann) == (-10, 3)


class TestCanvasLabels:
    @pytest.mark.parametrize("position", ["top", "bottom", "left", "right", "bottom-left"])
    def test_equilibrium_position_and_offset(self, position):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, label=Label(position=position, offset=8))
        _placed(_role(cvs.ax, "equilibrium_label")[0], position, 8)

    def test_equilibrium_label_without_text_keeps_default_text(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, label=Label(position="left"))
        assert _role(cvs.ax, "equilibrium_label")[0].get_text() == "$x^*$"

    def test_text_colour_and_size(self):
        cvs = Canvas().add_point(2, 2, label=Label(text="Q", color="#123456", fontsize=15))
        text = _role(cvs.ax, "point_label")[0]
        assert text.get_text() == "$Q$"
        assert to_hex(text.get_color()) == "#123456"
        assert text.get_fontsize() == 15

    def test_colour_follows_marker(self):
        cvs = Canvas().add_point(2, 2, label="P", marker=Marker(color="#aa0000"))
        assert to_hex(_role(cvs.ax, "point_label")[0].get_color()) == "#aa0000"
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, marker=Marker(color="#00aa00"))
        assert to_hex(_role(cvs.ax, "equilibrium_label")[0].get_color()) == "#00aa00"

    def test_label_colour_beats_marker_colour(self):
        cvs = Canvas().add_point(2, 2, label=Label(text="P", color="#0000aa"), marker=Marker(color="#aa0000"))
        assert to_hex(_role(cvs.ax, "point_label")[0].get_color()) == "#0000aa"

    def test_label_position_beats_legacy_offset(self):
        cvs = Canvas().add_point(2, 2, label=Label(text="P", position="left"), offset=(-10, 3))
        _placed(_role(cvs.ax, "point_label")[0], "left", 5)

    def test_hide(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ, label=Label(visible=False))
        assert not _role(cvs.ax, "equilibrium_label")[0].get_visible()

    def test_decomposition_labels_can_be_hidden(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, point_label=Label(visible=False))
        texts = _role(cvs.ax, "bundle_label")
        assert texts and not any(t.get_visible() for t in texts)

    def test_decomposition_labels_move(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, point_label=Label(position="bottom-left"))
        for text in _role(cvs.ax, "bundle_label"):
            _placed(text, "bottom-left", 6)

    def test_decomposition_default_offset(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC)
        assert all(tuple(t.xyann) == (6, 6) for t in _role(cvs.ax, "bundle_label"))

    def test_ic_labels(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(
            MODEL, levels=[3, 5], ic_label=Label(text="{:.1f}", position="top")
        )
        texts = _role(cvs.ax, "ic_label")
        assert [t.get_text() for t in texts] == ["3.0", "5.0"]
        _placed(texts[0], "top", 4)

    def test_ic_labels_default_right(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3], show_ic_labels=True)
        text = _role(cvs.ax, "ic_label")[0]
        assert text.get_text() == "3"
        _placed(text, "right", 4)

    def test_bliss_label(self):
        model = Satiation(bliss_x=6, bliss_y=4)
        cvs = Canvas(x_max=12, y_max=10).add_utility(model, levels=[-8], bliss_label="B")
        assert _role(cvs.ax, "bliss_label")[0].get_text() == "$B$"
        cvs = Canvas(x_max=12, y_max=10).add_utility(model, levels=[-8], bliss_label=Label(visible=False))
        text = _role(cvs.ax, "bliss_label")[0]
        assert text.get_text() == r"$\mathbf{x}^*$" and not text.get_visible()

    def test_label_reaches_tikz(self, tmp_path):
        path = tmp_path / "l.tex"
        Canvas().add_point(5, 5, label=Label(text="Q", position="left")).save(str(path))
        assert "Q" in path.read_text()


class TestOtherDiagrams:
    def test_demand_diagram(self):
        path = PricePath(
            MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px", price_range=(0.8, 6.0), n=20
        )
        fig = DemandDiagram(path).add_marshallian_panel(price_markers=[1.5, 4.0], point_label=Label(visible=False))
        texts = _role(fig.utility_canvas.ax, "equilibrium_label") + _role(fig.demand_canvas.ax, "point_label")
        assert len(texts) == 4 and not any(t.get_visible() for t in texts)

    def test_edgeworth(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0)
        box.add_endowment(7.0, 3.0, label=Label(text="\\omega", position="bottom"))
        box.add_walrasian_equilibrium(px=1.0, py=1.0, label=Label(color="#123456"))
        endowment = _role(box.ax, "endowment_label")[0]
        assert endowment.get_text() == "$\\omega$"
        _placed(endowment, "bottom", 5)
        walrasian = _role(box.ax, "walrasian_label")[0]
        assert walrasian.get_text() == "$X^*$"
        assert to_hex(walrasian.get_color()) == "#123456"

    def test_edgeworth_plain_strings(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0).add_endowment(7.0, 3.0, label="E")
        assert _role(box.ax, "endowment_label")[0].get_text() == "$E$"


def test_theme_defaults_apply():
    from econ_viz.themes.theme import Theme

    class Big(Theme):
        @property
        def point_label(self):
            return Label(position="left", offset=9, fontsize=20)

    cvs = Canvas(x_max=20, y_max=15, theme=Big(name="big")).add_equilibrium(EQ)
    text = _role(cvs.ax, "equilibrium_label")[0]
    assert text.get_fontsize() == 20
    _placed(text, "left", 9)


def test_label_is_exported():
    import econ_viz

    assert econ_viz.Label is Label

"""Tests for Effect: colour, position, and labels of decomposition effect arrows."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, Effect, LabelPosition
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import decompose_price_effect

DEC = decompose_price_effect(CobbDouglas(alpha=0.5, beta=0.5), px=(2.0, 4.0), py=3.0, income=30.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _canvas(**kwargs):
    return Canvas(x_max=16, y_max=12).add_decomposition(DEC, **kwargs)


class TestEffectValue:
    def test_defaults(self):
        effect = Effect()
        assert (effect.color, effect.y, effect.label) == (None, None, None)
        assert effect.label_position is LabelPosition.RIGHT

    def test_accepts_position_strings(self):
        assert Effect(label_position="below").label_position is LabelPosition.BOTTOM
        assert Effect(label_position="top").label_position is LabelPosition.TOP

    def test_rejects_unknown_position(self):
        with pytest.raises(InvalidParameterError, match="label_position"):
            Effect(label_position="middle")


class TestColour:
    def test_colour_applies_to_arrow_and_legend(self):
        cvs = _canvas(substitution=Effect(color="#123456"), income=Effect(color="#654321"))
        assert to_hex(_role(cvs.ax, "substitution")[0].arrow_patch.get_edgecolor()) == "#123456"
        assert to_hex(_role(cvs.ax, "income")[0].arrow_patch.get_edgecolor()) == "#654321"
        handles = {h._ev_role: h for h in cvs._legend_handles if getattr(h, "_ev_role", None)}
        assert to_hex(handles["substitution"].get_color()) == "#123456"

    def test_colour_applies_to_range_arrows(self):
        cvs = _canvas(show_x_projections=True, income=Effect(color="#654321"))
        ranges = _role(cvs.ax, "range")
        assert "#654321" in {to_hex(r.arrow_patch.get_edgecolor()) for r in ranges}


class TestPosition:
    def test_default_positions_are_unchanged(self):
        cvs = _canvas(show_x_projections=True)
        assert sorted(r.xy[1] for r in _role(cvs.ax, "range")) == pytest.approx([-0.15, -0.10])

    def test_y_moves_range_arrows_and_guides_follow(self):
        cvs = _canvas(show_x_projections=True, substitution=Effect(y=-0.06), income=Effect(y=-0.30))
        assert sorted(r.xy[1] for r in _role(cvs.ax, "range")) == pytest.approx([-0.30, -0.06])
        bottoms = [min(g.get_ydata()) for g in _role(cvs.ax, "guide")]
        assert all(b <= -0.30 for b in bottoms)


class TestLabels:
    def test_no_labels_by_default(self):
        cvs = _canvas()
        assert not _role(cvs.ax, "substitution_label")

    @pytest.mark.parametrize("projections", [False, True])
    def test_labels_in_both_modes(self, projections):
        cvs = _canvas(
            show_x_projections=projections,
            substitution=Effect(label="Substitution", color="#123456"),
            income=Effect(label="Income"),
        )
        sub = _role(cvs.ax, "substitution_label")
        inc = _role(cvs.ax, "income_label")
        assert [t.get_text() for t in sub] == ["Substitution"]
        assert [t.get_text() for t in inc] == ["Income"]
        assert to_hex(sub[0].get_color()) == "#123456"

    @pytest.mark.parametrize("position, ha, va", [
        ("top", "center", "bottom"),
        ("bottom", "center", "top"),
        ("left", "right", "center"),
        ("right", "left", "center"),
    ])
    def test_label_alignment_follows_position(self, position, ha, va):
        cvs = _canvas(substitution=Effect(label="S", label_position=position))
        label = _role(cvs.ax, "substitution_label")[0]
        assert (label.get_ha(), label.get_va()) == (ha, va)

    def test_left_and_right_labels_sit_outside_the_arrow(self):
        cvs = _canvas(
            show_x_projections=True,
            substitution=Effect(label="L", label_position="left"),
            income=Effect(label="R", label_position="right"),
        )
        left = _role(cvs.ax, "substitution_label")[0]
        right = _role(cvs.ax, "income_label")[0]
        assert left.xy[0] == pytest.approx(min(DEC.A.x, DEC.B.x))
        assert right.xy[0] == pytest.approx(max(DEC.B.x, DEC.C.x))

    def test_label_offset_moves_text(self):
        near = _role(_canvas(substitution=Effect(label="S", label_position="top", label_offset=2)).ax, "substitution_label")[0]
        far = _role(_canvas(substitution=Effect(label="S", label_position="top", label_offset=20)).ax, "substitution_label")[0]
        assert near.xyann[1] == pytest.approx(2)
        assert far.xyann[1] == pytest.approx(20)

    def test_labels_export_to_tikz(self, tmp_path):
        path = tmp_path / "effects.tex"
        _canvas(show_x_projections=True, substitution=Effect(label="Substitution")).save(str(path))
        assert "Substitution" in path.read_text()


def test_effect_colour_is_overridden_by_stroke_colour():
    from econ_viz import Stroke

    cvs = _canvas(substitution=Effect(color="#123456"), substitution_stroke=Stroke(color="#abcdef"))
    assert to_hex(_role(cvs.ax, "substitution")[0].arrow_patch.get_edgecolor()) == "#abcdef"


def test_in_plane_side_labels_sit_beside_the_middle():
    cvs = _canvas(substitution=Effect(label="S", label_position="right"))
    label = _role(cvs.ax, "substitution_label")[0]
    assert label.xy == pytest.approx(((DEC.A.x + DEC.B.x) / 2, (DEC.A.y + DEC.B.y) / 2))

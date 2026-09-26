"""Tests for #78: indifference-curve hierarchy and textbook labeling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from econ_viz import Canvas, Stroke, themes
from econ_viz.models import CobbDouglas, Leontief

MODEL = CobbDouglas(alpha=0.5, beta=0.5)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


class TestHighlightLevel:
    def test_no_highlight_draws_a_single_uniform_curve_set(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8])
        assert len(_role(cvs.ax, "curve")) == 1
        assert len(_role(cvs.ax, "secondary_curve")) == 0

    def test_highlight_splits_into_focal_and_secondary_curve_sets(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8], highlight_level=5)
        focal = _role(cvs.ax, "curve")
        secondary = _role(cvs.ax, "secondary_curve")
        assert len(focal) == 1
        assert len(secondary) == 1

    def test_secondary_curve_is_thinner_and_more_transparent_by_default(self):
        t = themes.default
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8], highlight_level=5)
        focal = _role(cvs.ax, "curve")[0]
        secondary = _role(cvs.ax, "secondary_curve")[0]
        assert secondary.get_linewidths()[0] < focal.get_linewidths()[0]
        assert secondary.get_alpha() is not None and secondary.get_alpha() < 1.0
        assert secondary.get_linewidths()[0] == pytest.approx(t.secondary_ic_linewidth)
        assert secondary.get_alpha() == pytest.approx(t.secondary_ic_opacity)

    def test_highlight_picks_the_nearest_available_level(self):
        # 5.3 isn't an explicit level; the curve nearest to it (5) becomes focal.
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8], highlight_level=5.3)
        assert len(_role(cvs.ax, "curve")) == 1
        assert len(_role(cvs.ax, "secondary_curve")) == 1

    def test_explicit_secondary_stroke_overrides_theme_default(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(
            MODEL, levels=[3, 5, 8], highlight_level=5, secondary_stroke=Stroke(width=2.5, opacity=0.2)
        )
        secondary = _role(cvs.ax, "secondary_curve")[0]
        assert secondary.get_linewidths()[0] == pytest.approx(2.5)
        assert secondary.get_alpha() == pytest.approx(0.2)

    def test_kinked_preferences_still_supported_with_highlight(self):
        model = Leontief(a=1.0, b=1.0)
        cvs = Canvas(x_max=20, y_max=15).add_utility(model, levels=[2, 4, 6], highlight_level=4, show_kinks=True)
        assert len(_role(cvs.ax, "curve")) == 1
        assert len(_role(cvs.ax, "secondary_curve")) == 1
        assert len(_role(cvs.ax, "kink")) > 0


class TestLabelStyle:
    def test_numeric_is_the_default_label_style(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8], show_ic_labels=True)
        texts = {a.get_text() for a in cvs.ax.texts if getattr(a, "_ev_role", None) == "ic_label"}
        assert texts == {"3", "5", "8"}

    def test_ordinal_labels_each_level_in_ascending_order(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(
            MODEL, levels=[3, 5, 8], show_ic_labels=True, label_style="ordinal"
        )
        texts = {
            a.get_text() for a in cvs.ax.texts if getattr(a, "_ev_role", None) in ("ic_label", "secondary_ic_label")
        }
        assert texts == {"$u_{1}$", "$u_{2}$", "$u_{3}$"}

    def test_inline_labels_avoid_clipping_past_the_visible_area(self):
        cvs = Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5, 8], show_ic_labels=True)
        for text in cvs.ax.texts:
            if getattr(text, "_ev_role", None) != "ic_label":
                continue
            x, _y = text.xy
            assert x < 20 * 0.97


class TestThemeControlsHierarchy:
    def test_secondary_ic_stroke_reflects_theme_fields(self):
        from dataclasses import replace

        custom = replace(
            themes.default, secondary_ic_color="#123456", secondary_ic_linewidth=2.2, secondary_ic_opacity=0.3
        )
        cvs = Canvas(x_max=20, y_max=15, theme=custom).add_utility(MODEL, levels=[3, 5, 8], highlight_level=5)
        secondary = _role(cvs.ax, "secondary_curve")[0]
        assert secondary.get_linewidths()[0] == pytest.approx(2.2)
        assert secondary.get_alpha() == pytest.approx(0.3)

    def test_secondary_ic_color_falls_back_to_ic_color(self):
        assert themes.default.secondary_ic_color is None
        assert themes.default.secondary_ic_stroke.color == themes.default.ic_color

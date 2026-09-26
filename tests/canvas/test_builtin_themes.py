"""Tests for #76: purpose-driven built-in themes (paper, monochrome, presentation, dark)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, themes
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.models import CobbDouglas

MODEL = CobbDouglas(alpha=0.5, beta=0.5)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


class TestThemesAreRegistered:
    @pytest.mark.parametrize("name", ["default", "colorblind", "nord", "paper", "monochrome", "presentation", "dark"])
    def test_available_from_themes_module(self, name):
        theme = getattr(themes, name)
        assert theme.name == name

    @pytest.mark.parametrize("name", ["paper", "monochrome", "presentation", "dark"])
    def test_resolvable_by_cli(self, name):
        from econ_viz.cli.resolve import resolve_theme

        assert resolve_theme(name).name == name

    @pytest.mark.parametrize("name", ["paper", "monochrome", "presentation", "dark"])
    def test_usable_as_config_base(self, name):
        from econ_viz import Config

        assert Config.from_dict({"base": name}).theme.name == name


class TestPaper:
    def test_thinner_and_smaller_than_default(self):
        default, paper = themes.default, themes.paper
        assert paper.ic_linewidth < default.ic_linewidth
        assert paper.budget_linewidth < default.budget_linewidth
        assert paper.eq_markersize < default.eq_markersize
        assert paper.label_scale < 1.0

    def test_transparent_background(self):
        cvs = Canvas(x_max=20, y_max=15, theme=themes.paper)
        assert cvs.ax.patch.get_alpha() == 0.0


class TestMonochromeIsReadableWithoutColour:
    def test_every_role_colour_is_a_shade_of_gray(self):
        for color in (
            themes.monochrome.ic_color,
            themes.monochrome.budget_color,
            themes.monochrome.path_color,
            themes.monochrome.eq_color,
            themes.monochrome.kink_color,
        ):
            r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
            assert r == g == b, f"{color} is not a neutral gray"

    def test_curves_budget_and_ray_use_different_line_styles(self):
        cvs = Canvas(x_max=20, y_max=15, theme=themes.monochrome).add_utility(MODEL, levels=[3]).add_budget(2, 3, 30)
        cvs.add_ray(0.5)
        curve = _role(cvs.ax, "curve")[0]
        budget = _role(cvs.ax, "budget")[0]
        ray = _role(cvs.ax, "ray")[0]
        styles = {curve.get_linestyle()[0][1], budget.get_linestyle(), ray.get_linestyle()}
        assert len(styles) == 3, f"expected three distinct line styles, got {styles}"

    def test_equilibrium_and_point_use_different_marker_shapes(self):
        from econ_viz import solve

        eq = solve(MODEL, px=2.0, py=3.0, income=30.0)
        cvs = Canvas(x_max=20, y_max=15, theme=themes.monochrome).add_equilibrium(eq).add_point(3, 2, label="A")
        eq_point = _role(cvs.ax, "equilibrium")[0]
        point = _role(cvs.ax, "point")[0]
        assert eq_point.get_marker() != point.get_marker()

    def test_decomposition_effect_arrows_differ_in_style(self):
        from econ_viz.optimizer import decompose_price_effect

        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        cvs = Canvas(x_max=20, y_max=15, theme=themes.monochrome).add_decomposition(dec)
        sub = _role(cvs.ax, "substitution")[0]
        inc = _role(cvs.ax, "income")[0]
        assert sub.arrow_patch.get_linestyle() != inc.arrow_patch.get_linestyle()


class TestPresentation:
    def test_larger_text_lines_and_markers_than_default(self):
        default, presentation = themes.default, themes.presentation
        assert presentation.label_scale > default.label_scale
        assert presentation.ic_linewidth > default.ic_linewidth
        assert presentation.budget_linewidth > default.budget_linewidth
        assert presentation.eq_markersize > default.eq_markersize
        assert presentation.point_label.fontsize > default.point_label.fontsize
        assert presentation.axis_label.fontsize > default.axis_label.fontsize

    def test_title_and_box_text_are_enlarged(self):
        assert themes.presentation.title_label.fontsize > 12
        assert themes.presentation.box_label.fontsize > 12
        cvs = Canvas(x_max=20, y_max=15, theme=themes.presentation, title="T")
        assert cvs.ax.title.get_fontsize() == pytest.approx(22)


class TestDark:
    def test_has_a_dark_background_and_light_foreground(self):
        theme = themes.dark
        assert theme.background_color is not None
        bg_r, bg_g, bg_b = (int(theme.background_color[i : i + 2], 16) for i in (1, 3, 5))
        assert (bg_r + bg_g + bg_b) / 3 < 100  # dark
        fg_r, fg_g, fg_b = (int(theme.label_color[i : i + 2], 16) for i in (1, 3, 5))
        assert (fg_r + fg_g + fg_b) / 3 > 180  # light, readable against the dark background

    def test_canvas_background_is_opaque_and_coloured(self):
        cvs = Canvas(x_max=20, y_max=15, theme=themes.dark)
        assert cvs.ax.patch.get_alpha() == 1.0
        assert to_hex(cvs.ax.patch.get_facecolor()) == to_hex(themes.dark.background_color)
        assert cvs.fig.patch.get_alpha() == 1.0

    def test_save_is_not_forced_transparent(self, tmp_path):
        path = tmp_path / "dark.png"
        Canvas(x_max=20, y_max=15, theme=themes.dark).add_budget(2, 3, 30).save(str(path))
        from matplotlib.image import imread

        pixel = imread(str(path))[0, 0]
        # A fully transparent corner would have alpha 0; the dark background should be opaque.
        assert pixel[-1] == pytest.approx(1.0)

    def test_default_theme_still_saves_transparent(self, tmp_path):
        path = tmp_path / "default.png"
        Canvas(x_max=20, y_max=15).add_budget(2, 3, 30).save(str(path))
        from matplotlib.image import imread

        pixel = imread(str(path))[0, 0]
        assert pixel[-1] == pytest.approx(0.0)

    def test_edgeworth_background_opt_in(self):
        default_box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0)
        assert default_box.ax.patch.get_alpha() != 1.0 or default_box.ax.patch.get_facecolor() != (
            *[int(themes.dark.background_color[i : i + 2], 16) / 255 for i in (1, 3, 5)],
            1.0,
        )
        dark_box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0, theme=themes.dark)
        assert to_hex(dark_box.ax.patch.get_facecolor()) == to_hex(themes.dark.background_color)
        assert dark_box.ax.patch.get_alpha() == 1.0


class TestBackgroundColorField:
    def test_none_keeps_transparent_default(self):
        theme = themes.default
        assert theme.background_color is None

    def test_custom_theme_background(self):
        from dataclasses import replace

        theme = replace(themes.default, background_color="#123456")
        cvs = Canvas(x_max=20, y_max=15, theme=theme)
        assert to_hex(cvs.ax.patch.get_facecolor()) == "#123456"
        assert to_hex(cvs.fig.patch.get_facecolor()) == "#123456"

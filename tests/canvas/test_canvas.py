"""Tests for Canvas, components, and Layer using the Agg (non-interactive) backend."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from econ_viz import ArrowStyle, Canvas, LabelPosition, LineStyle
from econ_viz.consumer.paths import PricePath, LinearBudget
from econ_viz.canvas.layers import Layer
from econ_viz.components import IndifferenceCurves, BudgetConstraint, EquilibriumPoint, draw_ray
from econ_viz.exceptions import ExportError, InvalidParameterError
from econ_viz.models import CobbDouglas, Leontief, Satiation, QuasiLinear
from econ_viz.optimizer import (
    DecompositionMethod,
    Equilibrium,
    decompose_price_effect,
    solve,
)
from econ_viz import themes


class TestCanvasInit:
    """Canvas construction and axis configuration."""

    def test_creates_figure(self):
        cvs = Canvas(x_max=10, y_max=10)
        assert cvs.fig is not None
        assert cvs.ax is not None

    def test_axis_limits(self):
        cvs = Canvas(x_max=15, y_max=12)
        assert cvs.ax.get_xlim() == (0, 15)
        assert cvs.ax.get_ylim() == (0, 12)

    def test_dpi_clamped_high(self):
        cvs = Canvas(dpi=9999)
        assert cvs.dpi == 1200

    def test_dpi_clamped_low(self):
        cvs = Canvas(dpi=0)
        assert cvs.dpi == 1

    def test_theme_stored(self):
        cvs = Canvas(theme=themes.nord)
        assert cvs.theme is themes.nord

    def test_label_pos_right_top(self):
        cvs = Canvas(x_label="Q", y_label="P", x_label_pos="right", y_label_pos="top")
        assert cvs.x_label_pos == "right"
        assert cvs.y_label_pos == "top"

    def test_label_pos_bottom_left(self):
        cvs = Canvas(x_label_pos="bottom", y_label_pos="left")
        assert cvs.x_label_pos is LabelPosition.BOTTOM
        assert cvs.y_label_pos is LabelPosition.LEFT

    @pytest.mark.parametrize(
        ("position", "offset", "alignment"),
        [
            (LabelPosition.TOP, (0, 8), ("center", "bottom")),
            (LabelPosition.RIGHT, (8, 0), ("left", "center")),
            (LabelPosition.BOTTOM, (0, -8), ("center", "top")),
        ],
    )
    def test_x_label_positions_around_arrow(self, position, offset, alignment):
        cvs = Canvas(x_max=10, x_label="Q", x_label_pos=position)
        label = next(text for text in cvs.ax.texts if getattr(text, "_ev_axis_label", None) == "x")

        assert label.xy == (10, 0)
        assert label.get_position() == offset
        assert (label.get_ha(), label.get_va()) == alignment

    @pytest.mark.parametrize(
        ("position", "offset", "alignment"),
        [
            (LabelPosition.LEFT, (-8, 0), ("right", "center")),
            (LabelPosition.TOP, (0, 8), ("center", "bottom")),
            (LabelPosition.RIGHT, (8, 0), ("left", "center")),
        ],
    )
    def test_y_label_positions_around_arrow(self, position, offset, alignment):
        cvs = Canvas(y_max=10, y_label="P", y_label_pos=position)
        label = next(text for text in cvs.ax.texts if getattr(text, "_ev_axis_label", None) == "y")

        assert label.xy == (0, 10)
        assert label.get_position() == offset
        assert (label.get_ha(), label.get_va()) == alignment

    def test_rejects_positions_that_do_not_surround_the_axis_arrow(self):
        with pytest.raises(ValueError, match="x-axis label"):
            Canvas(x_label_pos=LabelPosition.LEFT)
        with pytest.raises(ValueError, match="y-axis label"):
            Canvas(y_label_pos=LabelPosition.BOTTOM)

    @pytest.mark.parametrize("style", list(ArrowStyle))
    def test_axis_arrow_styles_are_selectable(self, style):
        cvs = Canvas(x_arrow_style=style, y_arrow_style=style)
        arrows = {
            getattr(patch, "_ev_axis_arrow", None): patch
            for patch in cvs.ax.patches
            if getattr(patch, "_ev_axis_arrow", None)
        }

        assert cvs.x_arrow_style is style
        assert cvs.y_arrow_style is style
        assert set(arrows) == {"x", "y"}
        assert arrows["x"]._ev_arrow_style is style
        assert arrows["y"]._ev_arrow_style is style

    def test_axis_arrow_styles_accept_values_and_reject_unknown_names(self):
        cvs = Canvas(x_arrow_style="->", y_arrow_style="wedge")
        assert cvs.x_arrow_style is ArrowStyle.SIMPLE
        assert cvs.y_arrow_style is ArrowStyle.WEDGE

        with pytest.raises(ValueError, match="arrow style"):
            Canvas(x_arrow_style="missing")

    @staticmethod
    def _axis_arrows(cvs):
        return {
            patch._ev_axis_arrow: patch
            for patch in cvs.ax.patches
            if getattr(patch, "_ev_axis_arrow", None)
        }

    @pytest.mark.parametrize("style", list(ArrowStyle))
    def test_axis_arrow_matches_spine_linewidth(self, style):
        cvs = Canvas(x_arrow_style=style, y_arrow_style=style)
        arrows = self._axis_arrows(cvs)
        assert arrows["x"].get_linewidth() == pytest.approx(cvs.ax.spines["bottom"].get_linewidth())
        assert arrows["y"].get_linewidth() == pytest.approx(cvs.ax.spines["left"].get_linewidth())

    @pytest.mark.parametrize("style", [ArrowStyle.SIMPLE, ArrowStyle.TRIANGLE, ArrowStyle.FANCY])
    def test_head_only_arrows_do_not_cover_the_spine(self, style):
        cvs = Canvas(x_max=10, y_max=10, x_arrow_style=style, y_arrow_style=style)
        arrows = self._axis_arrows(cvs)
        (x_start, _), _ = arrows["x"]._posA_posB
        (_, y_start), _ = arrows["y"]._posA_posB
        assert x_start > 9.95
        assert y_start > 9.95

    def test_wedge_arrow_keeps_a_visible_length(self):
        cvs = Canvas(x_max=10, x_arrow_style=ArrowStyle.WEDGE)
        (x_start, _), _ = self._axis_arrows(cvs)["x"]._posA_posB
        assert x_start < 9.8

    def test_axis_line_styles_default_to_solid(self):
        cvs = Canvas()
        assert cvs.x_line_style is LineStyle.SOLID
        assert cvs.y_line_style is LineStyle.SOLID
        assert cvs.ax.spines["bottom"].get_linestyle() == "solid"

    @pytest.mark.parametrize("style", [LineStyle.DASHED, LineStyle.DOTTED, LineStyle.DASHDOT])
    def test_axis_line_styles_apply_per_axis(self, style):
        cvs = Canvas(x_line_style=style)
        assert cvs.ax.spines["bottom"].get_linestyle() == style.value
        assert cvs.ax.spines["left"].get_linestyle() == "solid"

    def test_axis_line_styles_accept_values_and_reject_unknown_names(self):
        cvs = Canvas(y_line_style="dotted")
        assert cvs.y_line_style is LineStyle.DOTTED
        with pytest.raises(ValueError, match="line style"):
            Canvas(x_line_style="wavy")

    def test_dashed_axis_exports_to_tikz(self, tmp_path):
        path = tmp_path / "dashed.tex"
        Canvas(x_line_style=LineStyle.DASHED).save(str(path))
        dashed_lines = [line for line in path.read_text().splitlines() if "dash pattern=" in line or "dashed" in line]
        assert len(dashed_lines) == 1

    def test_title_set(self):
        cvs = Canvas(title="Test Title")
        assert cvs.title == "Test Title"


class TestCanvasAddUtility:
    """Canvas.add_utility() delegating to IndifferenceCurves component."""

    def test_returns_self(self):
        cvs = Canvas()
        result = cvs.add_utility(CobbDouglas(), levels=3)
        assert result is cvs

    def test_int_levels(self):
        Canvas(x_max=10, y_max=10).add_utility(CobbDouglas(), levels=3)

    def test_list_levels(self):
        Canvas(x_max=10, y_max=10).add_utility(CobbDouglas(), levels=[1.0, 2.0, 3.0])

    def test_leontief_rays_and_kinks(self):
        Canvas(x_max=10, y_max=10).add_utility(
            Leontief(), levels=3, show_rays=True, show_kinks=True
        )

    def test_custom_color(self):
        Canvas().add_utility(CobbDouglas(), levels=3, color="green")

    def test_smooth_show_rays_is_noop(self):
        """show_rays on a SMOOTH model must silently do nothing."""
        Canvas(x_max=10, y_max=10).add_utility(CobbDouglas(), levels=3, show_rays=True)

    def test_satiation(self):
        Canvas(x_max=15, y_max=15).add_utility(Satiation(bliss_x=7, bliss_y=7), levels=5)

    def test_quasi_linear(self):
        Canvas(x_max=15, y_max=15).add_utility(QuasiLinear(), levels=4)


class TestCanvasAddBudget:
    """Canvas.add_budget() delegating to BudgetConstraint component."""

    def test_returns_self(self):
        cvs = Canvas()
        result = cvs.add_budget(px=2, py=1, income=20)
        assert result is cvs

    def test_with_fill(self):
        Canvas(x_max=20, y_max=20).add_budget(px=2, py=3, income=30, fill=True)

    def test_with_label(self):
        Canvas(x_max=20, y_max=20).add_budget(px=2, py=3, income=30, label="BC")

    def test_custom_style(self):
        Canvas(x_max=20, y_max=20).add_budget(
            px=2, py=3, income=30, color="green", linewidth=2.0, linestyle="--"
        )

    def test_invalid_px_raises(self):
        with pytest.raises(InvalidParameterError):
            Canvas().add_budget(px=0, py=1, income=10)


class TestCanvasAddEquilibrium:
    """Canvas.add_equilibrium() delegating to EquilibriumPoint component."""

    def setup_method(self):
        self.eq = solve(CobbDouglas(), px=2.0, py=3.0, income=30.0)

    def test_returns_self(self):
        cvs = Canvas(x_max=20, y_max=20)
        assert cvs.add_equilibrium(self.eq) is cvs

    def test_with_ray(self):
        Canvas(x_max=20, y_max=20).add_equilibrium(self.eq, show_ray=True)

    def test_no_dashes(self):
        Canvas(x_max=20, y_max=20).add_equilibrium(self.eq, drop_dashes=False)

    def test_no_label(self):
        Canvas(x_max=20, y_max=20).add_equilibrium(self.eq, label=None)

    def test_custom_color(self):
        Canvas(x_max=20, y_max=20).add_equilibrium(self.eq, color="purple")


class TestCanvasAddRayAndPoint:
    """Canvas.add_ray() and Canvas.add_point() convenience methods."""

    def test_add_ray_returns_self(self):
        cvs = Canvas(x_max=10, y_max=10)
        assert cvs.add_ray(slope=1.0) is cvs

    def test_add_ray_steep(self):
        Canvas(x_max=10, y_max=10).add_ray(slope=5.0)

    def test_add_point_returns_self(self):
        cvs = Canvas(x_max=10, y_max=10)
        assert cvs.add_point(3.0, 4.0, label="A") is cvs

    def test_add_point_no_label(self):
        Canvas(x_max=10, y_max=10).add_point(2.0, 3.0)


class TestCanvasAddDecomposition:
    """Canvas.add_decomposition() integration tests."""

    def setup_method(self):
        model = CobbDouglas(alpha=0.5, beta=0.5)
        self.decomposition = decompose_price_effect(
            model,
            px=(2.0, 4.0),
            py=3.0,
            income=60.0,
            method=DecompositionMethod.SLUTSKY,
        )

    def test_returns_self(self):
        cvs = Canvas(x_max=35, y_max=25)
        assert cvs.add_decomposition(self.decomposition) is cvs

    def test_points_are_annotated(self):
        cvs = Canvas(x_max=35, y_max=25)
        cvs.add_decomposition(self.decomposition, show_arrows=False)
        labels = [txt.get_text() for txt in cvs.ax.texts]
        assert "$A$" in labels
        assert "$B$" in labels
        assert "$C$" in labels

    def test_overlapping_points_merge_labels(self):
        model = CobbDouglas(alpha=0.5, beta=0.5)
        same_price = decompose_price_effect(
            model,
            px=(2.0, 2.0),
            py=3.0,
            income=60.0,
            method=DecompositionMethod.SLUTSKY,
        )
        cvs = Canvas(x_max=35, y_max=25)
        cvs.add_decomposition(same_price, show_arrows=False)
        labels = [txt.get_text() for txt in cvs.ax.texts]
        assert "$A = B = C$" in labels

    def test_with_projection_labels(self):
        cvs = Canvas(x_max=35, y_max=25)
        cvs.add_decomposition(
            self.decomposition,
            show_arrows=True,
            label_effects=True,
            show_x_projections=True,
        )
        legend = cvs.ax.get_legend()
        assert legend is not None
        labels = [txt.get_text() for txt in legend.get_texts()]
        assert any("Sub:" in txt for txt in labels)
        assert any("Inc:" in txt for txt in labels)

    def test_save_with_decomposition(self, tmp_path):
        out = tmp_path / "decomposition.png"
        (
            Canvas(x_max=35, y_max=25)
            .add_utility(CobbDouglas(alpha=0.5, beta=0.5), levels=3)
            .add_decomposition(self.decomposition, show_x_projections=True)
            .save(str(out))
        )
        assert out.exists()


class TestCanvasAddPath:
    """Canvas.add_path() styling and geometry."""

    def test_default_path_color_differs_from_utility_curves(self):
        budget = LinearBudget(px=2.0, py=2.0, income=40.0)
        path = PricePath(CobbDouglas(), budget=budget, price="px", price_range=(1.0, 4.0), n=4)
        cvs = Canvas(x_max=25, y_max=25)

        cvs.add_utility(CobbDouglas(), levels=3)
        cvs.add_path(path)

        assert cvs.ax.lines[-1].get_color() == cvs.theme.path_color
        assert cvs.theme.path_color != cvs.theme.ic_color


class TestCanvasSave:
    """Canvas.save() format dispatch and error handling."""

    def test_save_png(self, tmp_path):
        out = str(tmp_path / "fig.png")
        Canvas().add_utility(CobbDouglas(), levels=3).save(out)
        assert (tmp_path / "fig.png").exists()

    def test_save_pdf(self, tmp_path):
        out = str(tmp_path / "fig.pdf")
        Canvas().save(out)
        assert (tmp_path / "fig.pdf").exists()

    def test_save_svg(self, tmp_path):
        out = str(tmp_path / "fig.svg")
        Canvas().save(out)
        assert (tmp_path / "fig.svg").exists()

    def test_save_unsupported_raises(self, tmp_path):
        with pytest.raises(ExportError):
            Canvas().save(str(tmp_path / "fig.bmp"))

    def test_save_tex(self, tmp_path):
        out = tmp_path / "fig.tex"
        Canvas().save(str(out))
        assert out.exists()
        assert r"\begin{tikzpicture}" in out.read_text(encoding="utf-8")

    def test_save_bad_path_raises(self):
        with pytest.raises(ExportError):
            Canvas().save("/nonexistent_dir/out.png")

    def test_method_chaining(self, tmp_path):
        """Full pipeline via method chaining must produce a file."""
        out = str(tmp_path / "chain.png")
        eq = solve(CobbDouglas(), 2.0, 3.0, 30.0)
        (Canvas(x_max=20, y_max=15)
            .add_utility(CobbDouglas(), levels=3)
            .add_budget(2.0, 3.0, 30.0, fill=True)
            .add_equilibrium(eq, show_ray=True)
            .save(out))
        assert (tmp_path / "chain.png").exists()


class TestCanvasThemes:
    """Canvas rendering under different themes."""

    def test_default_theme(self, tmp_path):
        Canvas(theme=themes.default).save(str(tmp_path / "default.png"))

    def test_nord_theme(self, tmp_path):
        Canvas(theme=themes.nord).save(str(tmp_path / "nord.png"))


class TestLayer:
    """Tests for Layer.compute_contour() mesh-grid generation."""

    def test_returns_correct_shapes(self):
        X, Y, Z = Layer.compute_contour(CobbDouglas(), (0.1, 10), (0.1, 10), res=50)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert Z.shape == (50, 50)

    def test_z_values_positive(self):
        _, _, Z = Layer.compute_contour(CobbDouglas(), (0.1, 5), (0.1, 5), res=20)
        assert np.all(Z > 0)

    def test_custom_res(self):
        X, _, _ = Layer.compute_contour(CobbDouglas(), (1, 10), (1, 10), res=100)
        assert X.shape == (100, 100)


class TestComponents:
    """Direct tests for the drawing component classes."""

    def setup_method(self):
        _, self.ax = plt.subplots()

    def test_budget_draw_no_fill(self):
        BudgetConstraint(px=2, py=3, income=30, color="blue", linewidth=1.5).draw(self.ax)

    def test_budget_draw_with_fill(self):
        BudgetConstraint(px=2, py=3, income=30, color="blue", linewidth=1.5,
                         fill=True, fill_alpha=0.1).draw(self.ax)

    def test_budget_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            BudgetConstraint(px=-1, py=1, income=10, color="blue", linewidth=1)

    def test_indifference_curves_int_levels(self):
        IndifferenceCurves(CobbDouglas(), levels=3, color="black",
                           linewidth=1.5).draw(self.ax, x_max=10, y_max=10)

    def test_indifference_curves_list_levels(self):
        IndifferenceCurves(CobbDouglas(), levels=[1.0, 2.0, 3.0], color="black",
                           linewidth=1.5).draw(self.ax, x_max=10, y_max=10)

    def test_indifference_curves_kinked_rays_kinks(self):
        IndifferenceCurves(Leontief(), levels=3, color="black", linewidth=1.5,
                           show_rays=True, show_kinks=True).draw(self.ax, x_max=10, y_max=10)

    def test_equilibrium_point_full(self):
        eq = Equilibrium(x=4.0, y=3.0, utility=2.0, bundle_type="interior")
        EquilibriumPoint(eq, color="red", drop_dashes=True,
                         show_ray=True, label="x^*").draw(self.ax, x_max=10, y_max=10)

    def test_equilibrium_point_no_label_no_dashes(self):
        eq = Equilibrium(x=4.0, y=3.0, utility=2.0, bundle_type="interior")
        EquilibriumPoint(eq, color="red", drop_dashes=False,
                         label=None).draw(self.ax, x_max=10, y_max=10)

    def test_equilibrium_point_x_near_zero_no_ray(self):
        """When x ≈ 0 the expansion-path ray must be suppressed even if show_ray=True."""
        eq = Equilibrium(x=0.0, y=5.0, utility=1.0, bundle_type="corner")
        EquilibriumPoint(eq, color="red", show_ray=True).draw(self.ax, x_max=10, y_max=10)

    def test_draw_ray_helper(self):
        draw_ray(self.ax, slope=1.0, x_max=10, y_max=10, color="black", linewidth=0.8)

    def test_draw_ray_steep_clips_to_y_max(self):
        """A slope steep enough to exceed y_max must be clipped correctly."""
        draw_ray(self.ax, slope=10.0, x_max=10, y_max=10, color="black", linewidth=0.8)


class TestLegendAndICLabels:
    """Tests for Canvas legend support and right-side IC labels (#11)."""

    def setup_method(self):
        self.cvs = Canvas(x_max=10, y_max=10)

    def test_add_utility_with_label_registers_handle(self):
        self.cvs.add_utility(CobbDouglas(), levels=2, label="$U_1$")
        assert len(self.cvs._legend_handles) == 1
        assert self.cvs._legend_handles[0].get_label() == "$U_1$"

    def test_add_utility_without_label_no_handle(self):
        self.cvs.add_utility(CobbDouglas(), levels=2)
        assert len(self.cvs._legend_handles) == 0

    def test_multiple_labels_accumulate(self):
        self.cvs.add_utility(CobbDouglas(), levels=2, label="$U_A$")
        self.cvs.add_utility(CobbDouglas(alpha=0.3, beta=0.7), levels=2, label="$U_B$")
        assert len(self.cvs._legend_handles) == 2

    def test_show_legend_returns_canvas(self):
        self.cvs.add_utility(CobbDouglas(), levels=2, label="$U_1$")
        result = self.cvs.show_legend()
        assert result is self.cvs

    def test_show_legend_creates_legend_object(self):
        self.cvs.add_utility(CobbDouglas(), levels=2, label="$U_1$")
        self.cvs.show_legend()
        assert self.cvs.ax.get_legend() is not None

    def test_show_legend_no_labels_no_legend(self):
        self.cvs.add_utility(CobbDouglas(), levels=2)
        self.cvs.show_legend()
        assert self.cvs.ax.get_legend() is None

    def test_show_ic_labels_does_not_raise(self):
        """IC labels must render without error."""
        self.cvs.add_utility(CobbDouglas(), levels=3, show_ic_labels=True)

    def test_show_ic_labels_custom_fmt(self):
        self.cvs.add_utility(CobbDouglas(), levels=2, show_ic_labels=True,
                              ic_label_fmt="{:.3f}")

    def test_chaining_with_legend(self):
        result = (
            self.cvs
            .add_utility(CobbDouglas(), levels=2, label="$U_1$")
            .show_legend()
        )
        assert result is self.cvs

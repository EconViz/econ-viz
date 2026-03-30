"""Tests for Canvas, components, and Layer using the Agg (non-interactive) backend."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from econ_viz import Canvas
from econ_viz.canvas.layers import Layer
from econ_viz.components import IndifferenceCurves, BudgetConstraint, EquilibriumPoint, draw_ray
from econ_viz.exceptions import ExportError, InvalidParameterError
from econ_viz.models import CobbDouglas, Leontief, Satiation, QuasiLinear
from econ_viz.optimizer import Equilibrium, solve
from econ_viz import themes


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


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
        assert cvs.x_label_pos == "bottom"
        assert cvs.y_label_pos == "left"

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

    def test_save_tex_raises(self, tmp_path):
        with pytest.raises(ExportError):
            Canvas().save(str(tmp_path / "fig.tex"))

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

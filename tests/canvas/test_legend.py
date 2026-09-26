"""Tests for Legend: position, automatic placement, look, and visibility."""

import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from econ_viz import Canvas, DemandDiagram, Legend, LinearBudget, PricePath
from econ_viz.canvas.legend import place_legend
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.enums import LegendPosition
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas, Haagsma
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
DEC = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _renderer(cvs):
    return cvs.fig.canvas.get_renderer()


def _legend_box(cvs):
    return cvs.ax.get_legend().get_window_extent(_renderer(cvs))


def _axes_box(cvs):
    return cvs.ax.get_window_extent(_renderer(cvs))


def _tight_without_legend(ax, renderer):
    """The axes' tight bbox, which otherwise includes the legend itself."""
    legend = ax.get_legend()
    legend.set_visible(False)
    try:
        return ax.get_tightbbox(renderer)
    finally:
        legend.set_visible(True)


class TestLegendValue:
    def test_defaults(self):
        legend = Legend()
        assert (legend.position, legend.fontsize, legend.frame, legend.columns, legend.visible) == (
            None, None, None, None, None)

    def test_position_accepts_strings(self):
        assert Legend(position="bottom").position is LegendPosition.BOTTOM
        assert Legend(position="upper left").position is LegendPosition.UPPER_LEFT

    @pytest.mark.parametrize("kwargs", [
        {"position": "middle"}, {"fontsize": 0}, {"columns": 0}, {"columns": 1.5},
    ])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Legend(**kwargs)

    def test_merged_over_keeps_unset_fields(self):
        merged = Legend(fontsize=9).merged_over(Legend(position="top", fontsize=11, frame=True))
        assert (merged.position, merged.fontsize, merged.frame) == (LegendPosition.TOP, 9, True)

    def test_theme_default(self):
        t = Canvas().theme.legend
        assert (t.position, t.fontsize, t.frame, t.visible) == (LegendPosition.AUTO, 11, False, True)


class TestPositions:
    @pytest.mark.parametrize("position", ["upper right", "upper left", "lower left", "lower right"])
    def test_inside_corners(self, position):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(position=position))
        legend, axes = _legend_box(cvs), _axes_box(cvs)
        assert axes.x0 <= legend.x0 and legend.x1 <= axes.x1
        assert axes.y0 <= legend.y0 and legend.y1 <= axes.y1
        vertical, horizontal = position.split()
        assert (legend.y0 + legend.y1) / 2 > (axes.y0 + axes.y1) / 2 if vertical == "upper" else True
        assert (legend.x0 + legend.x1) / 2 < (axes.x0 + axes.x1) / 2 if horizontal == "left" else True

    @pytest.mark.parametrize("position, side", [
        ("top", "above"), ("bottom", "below"), ("left", "left"), ("right", "right"),
    ])
    def test_outside_positions_clear_the_axes(self, position, side):
        cvs = Canvas(x_max=20, y_max=15, title="T").add_decomposition(
            DEC, show_x_projections=True, legend=Legend(position=position))
        legend = _legend_box(cvs)
        tight = _tight_without_legend(cvs.ax, _renderer(cvs))
        if side == "above":
            assert legend.y0 >= tight.y1
        elif side == "below":
            assert legend.y1 <= tight.y0
        elif side == "left":
            assert legend.x1 <= tight.x0
        else:
            assert legend.x0 >= tight.x1

    def test_top_and_bottom_use_several_columns(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(position="bottom"))
        assert cvs.ax.get_legend()._ncols == 3
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(position="bottom", columns=5))
        assert cvs.ax.get_legend()._ncols == 5

    @pytest.mark.parametrize("position", ["bottom", "right"])
    def test_outside_legend_fits_in_saved_png(self, tmp_path, position):
        from matplotlib.image import imread

        def size(legend):
            path = tmp_path / f"{position}-{legend is not None}.png"
            Canvas(x_max=20, y_max=15, dpi=100).add_decomposition(DEC, legend=legend).save(str(path))
            height, width = imread(str(path)).shape[:2]
            return width, height

        width, height = size(Legend(position=position))
        hidden_width, hidden_height = size(Legend(visible=False))
        # The tight export grows to take in the outside legend.
        if position == "right":
            assert width > hidden_width
        else:
            assert height > hidden_height

    def test_outside_legend_reaches_tikz(self, tmp_path):
        path = tmp_path / "l.tex"
        Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(position="bottom")).save(str(path))
        assert "Sub" in path.read_text()


class TestAutomaticPlacement:
    def _haagsma(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*(inferior|Giffen) good.*")
            return decompose_price_effect(Haagsma(), px=(2.0, 1.0), py=1.0, income=28.0, method="hicks")

    def test_avoids_covering_the_diagram(self):
        # The fixed upper-right legend used to cover U1 here (#121).
        cvs = Canvas(x_max=16, y_max=32).add_decomposition(self._haagsma())
        legend = cvs.ax.get_legend()
        box = legend.get_window_extent(_renderer(cvs))
        renderer = _renderer(cvs)
        for cs in [a for a in cvs.ax.get_children() if getattr(a, "_ev_role", None) == "curve"]:
            for path in cs.get_paths():
                assert not cs.get_transform().transform_path(path).intersects_bbox(box, filled=False)

    def test_moves_outside_when_every_corner_is_covered(self):
        cvs = Canvas(x_max=10, y_max=10)
        for y in (1, 3, 5, 7, 9):
            cvs.ax.plot([0, 10], [y, y], label=f"y={y}")
        cvs.show_legend()
        box = _legend_box(cvs)
        assert box.x0 >= _tight_without_legend(cvs.ax, _renderer(cvs)).x1

    def test_explicit_position_is_kept(self):
        cvs = Canvas(x_max=16, y_max=32).add_decomposition(self._haagsma(), legend=Legend(position="upper right"))
        box, axes = _legend_box(cvs), _axes_box(cvs)
        assert box.x1 > (axes.x0 + axes.x1) / 2 and box.y1 > (axes.y0 + axes.y1) / 2


class TestLookAndVisibility:
    def test_fontsize_and_frame(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(fontsize=8, frame=True))
        legend = cvs.ax.get_legend()
        assert legend.get_texts()[0].get_fontsize() == 8
        assert legend.get_frame_on()

    def test_hidden(self):
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(DEC, legend=Legend(visible=False))
        assert cvs.ax.get_legend() is None

    def test_matplotlib_kwargs_still_work(self):
        cvs = Canvas().add_budget(2, 3, 30, label="B")
        cvs.show_legend(loc="lower left", fontsize=7)
        legend = cvs.ax.get_legend()
        assert legend._loc == 3 and legend.get_texts()[0].get_fontsize() == 7

    def test_empty_legend_draws_nothing(self):
        cvs = Canvas()
        assert place_legend(cvs.ax, [], [], Legend()) is None
        assert cvs.show_legend().ax.get_legend() is None


class TestOtherDiagrams:
    def test_demand_diagram(self):
        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=20)
        fig = DemandDiagram(path).add_marshallian_panel(price_markers=[1.5, 4.0], legend=Legend(position="bottom"))
        for cvs in (fig.utility_canvas, fig.demand_canvas):
            box = cvs.ax.get_legend().get_window_extent(_renderer(cvs))
            assert box.y1 <= _tight_without_legend(cvs.ax, _renderer(cvs)).y0

    def test_demand_diagram_hidden(self):
        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=20)
        fig = DemandDiagram(path).add_marshallian_panel(legend=Legend(visible=False))
        assert fig.utility_canvas.ax.get_legend() is None
        assert fig.demand_canvas.ax.get_legend() is None

    def test_edgeworth(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0).add_endowment(7.0, 3.0)
        box.add_contract_curve().add_price_line(px=1.0, py=1.0)
        box.show_legend(legend=Legend(position="right"))
        legend = box.ax.get_legend()
        assert legend is not None and legend.get_texts()[0].get_fontsize() == 10
        renderer = box.fig.canvas.get_renderer()
        assert legend.get_window_extent(renderer).x0 >= _tight_without_legend(box.ax, renderer).x1


def test_legend_is_exported():
    import econ_viz

    assert econ_viz.Legend is Legend

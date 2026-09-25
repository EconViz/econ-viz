"""Tests for Marker: colour, size, and shape of every point marker."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, DemandDiagram, LinearBudget, Marker, PricePath, solve
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas, Leontief, Satiation
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
EQ = solve(MODEL, px=2.0, py=3.0, income=30.0)
SQUARE = Marker(color="#123456", size=9.0, shape="s")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _looks_like(point, marker):
    if marker.color is not None:
        assert to_hex(point.get_markerfacecolor()) == to_hex(marker.color)
    if marker.size is not None:
        assert point.get_markersize() == pytest.approx(marker.size)
    if marker.shape is not None:
        assert point.get_marker() == marker.shape


class TestMarkerValue:
    def test_defaults(self):
        assert (Marker().color, Marker().size, Marker().shape) == (None, None, None)

    @pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": -2}, {"shape": "not-a-marker"}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Marker(**kwargs)

    def test_merged_over_keeps_unset_fields(self):
        merged = Marker(size=3).merged_over(Marker(color="red", size=9, shape="s"))
        assert (merged.color, merged.size, merged.shape) == ("red", 3, "s")


class TestThemeDefaults:
    """Without a Marker, every point is drawn exactly as its theme default says."""

    def test_canvas_points(self):
        cvs = Canvas(x_max=20, y_max=15).add_equilibrium(EQ).add_point(2, 2, label="P")
        t = cvs.theme
        _looks_like(_role(cvs.ax, "equilibrium")[0], t.eq_marker)
        _looks_like(_role(cvs.ax, "point")[0], t.point_marker)

    def test_kink_and_bliss(self):
        t = Canvas().theme
        cvs = Canvas().add_utility(Leontief(a=1, b=2), levels=[2, 4], show_kinks=True)
        _looks_like(_role(cvs.ax, "kink")[0], t.kink_marker)
        cvs = Canvas(x_max=12, y_max=10).add_utility(Satiation(bliss_x=6, bliss_y=4), levels=[-8, -2])
        bliss = _role(cvs.ax, "bliss")[0]
        _looks_like(bliss, t.bliss_marker)
        assert to_hex(bliss.get_markerfacecolor()) == to_hex(t.ic_color)

    def test_decomposition_bundles(self):
        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(dec)
        for point in _role(cvs.ax, "bundle"):
            _looks_like(point, cvs.theme.eq_marker)

    def test_path_points(self):
        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=5)
        cvs = Canvas(x_max=30, y_max=25).add_path(path, show_points=True)
        points = _role(cvs.ax, "path_point")
        assert len(points) == 5
        _looks_like(points[0], cvs.theme.path_marker)
        assert to_hex(points[0].get_markerfacecolor()) == to_hex(cvs.theme.path_color)


class TestCanvasMarkers:
    def test_point_and_equilibrium(self):
        cvs = Canvas(x_max=20, y_max=15).add_point(2, 2, marker=SQUARE).add_equilibrium(EQ, marker=SQUARE)
        _looks_like(_role(cvs.ax, "point")[0], SQUARE)
        _looks_like(_role(cvs.ax, "equilibrium")[0], SQUARE)

    def test_kink_and_bliss(self):
        cvs = Canvas().add_utility(Leontief(a=1, b=2), levels=[2, 4], show_kinks=True, kink_marker=SQUARE)
        for kink in _role(cvs.ax, "kink"):
            _looks_like(kink, SQUARE)
        cvs = Canvas(x_max=12, y_max=10).add_utility(
            Satiation(bliss_x=6, bliss_y=4), levels=[-8, -2], bliss_marker=SQUARE)
        _looks_like(_role(cvs.ax, "bliss")[0], SQUARE)

    def test_decomposition_bundles_and_legend(self):
        dec = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
        cvs = Canvas(x_max=20, y_max=15).add_decomposition(dec, point_marker=SQUARE)
        assert len(_role(cvs.ax, "bundle")) == 3
        for point in _role(cvs.ax, "bundle"):
            _looks_like(point, SQUARE)
        bundle_handles = [h for h in cvs._legend_handles if getattr(h, "_ev_role", None) == "bundle"]
        assert len(bundle_handles) == 3
        for handle in bundle_handles:
            _looks_like(handle, SQUARE)

    def test_path_points_and_equilibria(self):
        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=4)
        cvs = Canvas(x_max=30, y_max=25).add_path(path, show_points=True, point_marker=SQUARE)
        _looks_like(_role(cvs.ax, "path_point")[0], SQUARE)
        cvs = Canvas(x_max=30, y_max=25).add_path(path, show_equilibria=True, equilibrium_marker=SQUARE)
        _looks_like(_role(cvs.ax, "equilibrium")[0], SQUARE)

    def test_marker_overrides_plain_arguments(self):
        cvs = Canvas().add_point(2, 2, color="red", markersize=4, marker=Marker(size=11))
        point = _role(cvs.ax, "point")[0]
        assert point.get_markersize() == pytest.approx(11)
        assert to_hex(point.get_markerfacecolor()) == to_hex("red")

    def test_marker_shape_reaches_tikz(self, tmp_path):
        def tex(marker):
            path = tmp_path / "m.tex"
            Canvas().add_point(5, 5, marker=marker).save(str(path))
            return path.read_text()

        assert tex(Marker(shape="o")) != tex(Marker(shape="s"))


class TestOtherDiagrams:
    def test_demand_diagram_points(self):
        path = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                         price_range=(0.8, 6.0), n=20)
        fig = DemandDiagram(path).add_marshallian_panel(price_markers=[1.5, 4.0], point_marker=SQUARE)
        for point in _role(fig.utility_canvas.ax, "equilibrium") + _role(fig.demand_canvas.ax, "point"):
            _looks_like(point, SQUARE)

    def test_edgeworth_points(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0)
        box.add_endowment(7.0, 3.0, marker=SQUARE)
        box.add_walrasian_equilibrium(px=1.0, py=1.0, marker=SQUARE)
        _looks_like(_role(box.ax, "endowment")[0], SQUARE)
        _looks_like(_role(box.ax, "walrasian")[0], SQUARE)

    def test_edgeworth_walrasian_accepts_legacy_shape_string(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0).add_endowment(7.0, 3.0)
        box.add_walrasian_equilibrium(px=1.0, py=1.0, marker="D")
        assert _role(box.ax, "walrasian")[0].get_marker() == "D"

    def test_edgeworth_walrasian_marker_without_shape_keeps_star(self):
        box = EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0).add_endowment(7.0, 3.0)
        box.add_walrasian_equilibrium(px=1.0, py=1.0, marker=Marker(color="#123456"))
        assert _role(box.ax, "walrasian")[0].get_marker() == "*"


def test_marker_is_exported():
    import econ_viz

    assert econ_viz.Marker is Marker

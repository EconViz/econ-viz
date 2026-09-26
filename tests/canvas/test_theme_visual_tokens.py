"""Tests for #75: previously hard-coded visual literals now read from Theme.

Each test checks that a custom theme, applied *without* passing an explicit
method argument, actually changes what gets drawn -- the gap the issue is
about -- and that the built-in default's exact output is unchanged.
"""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, themes
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.models import CobbDouglas, Satiation, StoneGeary

MODEL = CobbDouglas(alpha=0.5, beta=0.5)


def _role(ax, role):
    return [a for a in ax.get_children() if getattr(a, "_ev_role", None) == role]


def _box(theme=None, **kwargs):
    kwargs.setdefault("total_x", 10.0)
    kwargs.setdefault("total_y", 10.0)
    if theme is not None:
        kwargs["theme"] = theme
    return EdgeworthBox(MODEL, MODEL, **kwargs)


class TestEdgeworthDefaultsUnchanged:
    """Pin the exact pre-#75 literal values so the theme wiring doesn't shift default output."""

    def test_core(self):
        box = _box().add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[6.0, 4.0], [7.0, 3.0], [8.0, 2.0]])
        box.add_core(min_points=1)
        point = _role(box.ax, "core")[0]
        assert to_hex(point.get_color()) == to_hex("#C0392B")
        assert point.get_linewidth() == pytest.approx(3.0)

    def test_price_line(self):
        box = _box().add_endowment(7.0, 3.0).add_price_line(px=1.0, py=1.0)
        line = _role(box.ax, "price")[0]
        assert to_hex(line.get_color()) == to_hex("#000000")
        assert line.get_linewidth() == pytest.approx(1.2)
        assert line.get_linestyle() == "--"

    def test_contract_curve(self):
        box = _box().add_contract_curve()
        line = _role(box.ax, "contract")[0]
        assert to_hex(line.get_color()) == to_hex("#000000")
        assert line.get_linewidth() == pytest.approx(1.2)
        assert line.get_linestyle() == "--"

    def test_walrasian_equilibrium(self):
        box = _box().add_endowment(7.0, 3.0).add_walrasian_equilibrium(px=1.0, py=1.0)
        point = _role(box.ax, "walrasian")[0]
        assert to_hex(point.get_markerfacecolor()) == to_hex("#2E86AB")
        assert point.get_markersize() == pytest.approx(10.0)
        assert point.get_marker() == "*"

    def test_endowment(self):
        box = _box().add_endowment(7.0, 3.0)
        point = _role(box.ax, "endowment")[0]
        assert point.get_markersize() == pytest.approx(max(box.theme.eq_markersize, 6.0))

    def test_bliss_marker_size(self):
        cvs = Canvas(x_max=12, y_max=10).add_utility(Satiation(bliss_x=6, bliss_y=4), levels=[-8])
        bliss = _role(cvs.ax, "bliss")[0]
        assert bliss.get_markersize() == pytest.approx(12.0)

    def test_subsistence_lines(self):
        model = StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0)
        cvs = Canvas(x_max=20, y_max=15).add_utility(model, levels=[1, 2])
        lines = _role(cvs.ax, "subsistence")
        assert len(lines) == 2
        for line in lines:
            assert to_hex(line.get_color()) == to_hex("gray")
            assert line.get_linewidth() == pytest.approx(0.8)


class TestCustomThemePropagates:
    """A custom Theme, with no explicit method argument, actually changes the output."""

    def test_core_color_and_width(self):
        theme = replace(themes.default, core_color="#00FF00", core_linewidth=5.0)
        box = _box(theme=theme).add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[6.0, 4.0], [7.0, 3.0], [8.0, 2.0]])
        box.add_core(min_points=1)
        point = _role(box.ax, "core")[0]
        assert to_hex(point.get_color()) == to_hex("#00FF00")
        assert point.get_linewidth() == pytest.approx(5.0)

    def test_core_point_marker(self):
        theme = replace(themes.default, core_color="#00FF00")
        box = _box(theme=theme).add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[7.0, 3.0]])
        box.add_core(min_points=2)
        point = _role(box.ax, "core_point")[0]
        assert to_hex(point.get_color()) == to_hex("#00FF00")

    def test_price_line_color_and_width(self):
        theme = replace(themes.default, price_color="#FF00FF", price_linewidth=4.0)
        box = _box(theme=theme).add_endowment(7.0, 3.0).add_price_line(px=1.0, py=1.0)
        line = _role(box.ax, "price")[0]
        assert to_hex(line.get_color()) == to_hex("#FF00FF")
        assert line.get_linewidth() == pytest.approx(4.0)

    def test_contract_curve_color_and_width(self):
        theme = replace(themes.default, contract_color="#123456", contract_linewidth=3.0)
        box = _box(theme=theme).add_contract_curve()
        line = _role(box.ax, "contract")[0]
        assert to_hex(line.get_color()) == to_hex("#123456")
        assert line.get_linewidth() == pytest.approx(3.0)

    def test_walrasian_color_and_size(self):
        theme = replace(themes.default, walrasian_color="#654321", walrasian_markersize=20.0)
        box = _box(theme=theme).add_endowment(7.0, 3.0).add_walrasian_equilibrium(px=1.0, py=1.0)
        point = _role(box.ax, "walrasian")[0]
        assert to_hex(point.get_markerfacecolor()) == to_hex("#654321")
        assert point.get_markersize() == pytest.approx(20.0)

    def test_endowment_marker_floor(self):
        theme = replace(themes.default, eq_markersize=2.0)
        box = _box(theme=theme).add_endowment(7.0, 3.0)
        point = _role(box.ax, "endowment")[0]
        assert point.get_markersize() == pytest.approx(6.0)

    def test_bliss_marker_size(self):
        from econ_viz import Marker
        from econ_viz.themes.theme import Theme

        class BigBliss(Theme):
            @property
            def bliss_marker(self):
                return Marker(size=25.0, shape="*")

        theme = BigBliss(name="big-bliss")
        cvs = Canvas(x_max=12, y_max=10, theme=theme).add_utility(Satiation(bliss_x=6, bliss_y=4), levels=[-8])
        bliss = _role(cvs.ax, "bliss")[0]
        assert bliss.get_markersize() == pytest.approx(25.0)

    def test_subsistence_lines_follow_theme(self):
        theme = replace(themes.default, subsistence_color="#ABCDEF", subsistence_linewidth=2.5)
        model = StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0)
        cvs = Canvas(x_max=20, y_max=15, theme=theme).add_utility(model, levels=[1, 2])
        lines = _role(cvs.ax, "subsistence")
        assert len(lines) == 2
        for line in lines:
            assert to_hex(line.get_color()) == to_hex("#ABCDEF")
            assert line.get_linewidth() == pytest.approx(2.5)


class TestExplicitArgumentStillWins:
    def test_add_core_explicit_color(self):
        box = _box().add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[6.0, 4.0], [7.0, 3.0], [8.0, 2.0]])
        box.add_core(color="#101010", min_points=1)
        point = _role(box.ax, "core")[0]
        assert to_hex(point.get_color()) == to_hex("#101010")

    def test_add_walrasian_explicit_shape_string(self):
        box = _box().add_endowment(7.0, 3.0)
        box.add_walrasian_equilibrium(px=1.0, py=1.0, marker="D")
        assert _role(box.ax, "walrasian")[0].get_marker() == "D"


class TestConfigOverridesReachNewRoles:
    def test_core_and_price_via_config(self):
        from econ_viz import Config

        config = Config.from_dict(
            {
                "stroke": {"core": {"color": "#0F0F0F"}, "price": {"color": "#F0F0F0"}},
                "marker": {"walrasian": {"color": "#ABABAB"}},
            }
        )
        box = _box(theme=config.theme).add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[6.0, 4.0], [7.0, 3.0], [8.0, 2.0]])
        box.add_core(min_points=1)
        box.add_price_line(px=1.0, py=1.0)
        box.add_walrasian_equilibrium(px=1.0, py=1.0)
        assert to_hex(_role(box.ax, "core")[0].get_color()) == to_hex("#0F0F0F")
        assert to_hex(_role(box.ax, "price")[0].get_color()) == to_hex("#F0F0F0")
        assert to_hex(_role(box.ax, "walrasian")[0].get_markerfacecolor()) == to_hex("#ABABAB")

    def test_subsistence_and_core_point_via_config(self):
        from econ_viz import Config

        config = Config.from_dict(
            {
                "stroke": {"subsistence": {"color": "#00AA00"}},
                "marker": {"core": {"color": "#AA0000"}},
            }
        )
        model = StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0)
        cvs = Canvas(x_max=20, y_max=15, theme=config.theme).add_utility(model, levels=[1, 2])
        for line in _role(cvs.ax, "subsistence"):
            assert to_hex(line.get_color()) == to_hex("#00AA00")

        box = _box(theme=config.theme).add_endowment(7.0, 3.0)
        box.contract_curve_points = np.array([[7.0, 3.0]])
        box.add_core(min_points=2)
        point = _role(box.ax, "core_point")[0]
        assert to_hex(point.get_color()) == to_hex("#AA0000")

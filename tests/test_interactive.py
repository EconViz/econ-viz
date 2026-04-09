"""Tests for the econ_viz.interactive submodule (issue #45).

Since ipywidgets and IPython are optional runtime dependencies, the tests
that inspect widget internals use mocks where full Jupyter integration is
not available.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from econ_viz import Canvas
from econ_viz.models import CobbDouglas


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ------------------------------------------------------------------
# WidgetViewer initialisation
# ------------------------------------------------------------------

class TestWidgetViewerInit:
    """WidgetViewer constructor validation."""

    def test_no_slider_specs_raises(self):
        from econ_viz.interactive import WidgetViewer

        with pytest.raises(ValueError, match="at least one slider spec"):
            WidgetViewer(lambda **kw: Canvas())

    def test_stores_specs(self):
        from econ_viz.interactive import WidgetViewer

        def _draw(p1: float) -> Canvas:
            return Canvas()

        viewer = WidgetViewer(_draw, p1=(1.0, 10.0, 0.5))
        assert "p1" in viewer._slider_specs
        assert viewer._slider_specs["p1"] == (1.0, 10.0, 0.5)

    def test_multiple_specs(self):
        from econ_viz.interactive import WidgetViewer

        viewer = WidgetViewer(
            lambda p1, income, **kw: Canvas(),
            p1=(1.0, 8.0, 0.5),
            income=(10.0, 50.0, 5.0),
        )
        assert len(viewer._slider_specs) == 2


# ------------------------------------------------------------------
# Slider builder
# ------------------------------------------------------------------

class TestBuildSliders:
    """_build_sliders() logic — requires ipywidgets at runtime."""

    @pytest.fixture()
    def viewer(self):
        from econ_viz.interactive import WidgetViewer
        return WidgetViewer(lambda p1: Canvas(), p1=(2.0, 10.0, 2.0))

    def test_slider_count(self, viewer):
        ipywidgets = pytest.importorskip("ipywidgets")
        sliders = viewer._build_sliders(ipywidgets)
        assert len(sliders) == 1
        assert "p1" in sliders

    def test_slider_mid_value(self, viewer):
        ipywidgets = pytest.importorskip("ipywidgets")
        sliders = viewer._build_sliders(ipywidgets)
        # mid of (2, 10) snapped to step=2 → 6.0
        assert sliders["p1"].value == 6.0

    def test_slider_bounds(self, viewer):
        ipywidgets = pytest.importorskip("ipywidgets")
        sliders = viewer._build_sliders(ipywidgets)
        assert sliders["p1"].min == 2.0
        assert sliders["p1"].max == 10.0
        assert sliders["p1"].step == 2.0


# ------------------------------------------------------------------
# Dependency guard
# ------------------------------------------------------------------

class TestRequireWidgets:
    """_require_widgets() yields clear ImportError messages."""

    def test_missing_ipywidgets(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "ipywidgets":
                raise ImportError("no ipywidgets")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        from econ_viz.interactive.widgets import _require_widgets

        with pytest.raises(ImportError, match="ipywidgets"):
            _require_widgets()

    def test_missing_ipython(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "IPython":
                raise ImportError("no IPython")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        from econ_viz.interactive.widgets import _require_widgets

        with pytest.raises(ImportError, match="IPython"):
            _require_widgets()


# ------------------------------------------------------------------
# show() guard
# ------------------------------------------------------------------

class TestWidgetViewerShow:
    """WidgetViewer.show() raises ImportError when deps are missing."""

    def test_show_raises_without_deps(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name in ("ipywidgets", "IPython"):
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        from econ_viz.interactive import WidgetViewer
        viewer = WidgetViewer(lambda p1: Canvas(), p1=(1.0, 5.0, 1.0))

        with pytest.raises(ImportError):
            viewer.show()

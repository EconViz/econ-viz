"""Regression tests for the pure-TikZ export backend."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

from econ_viz import Canvas
from econ_viz.io import save_figure
from econ_viz.io.backend_tikz import TikzRenderer
from econ_viz.io.backend_tikz._document import assemble
from econ_viz.io.backend_tikz._path import dash_spec, path_to_polylines, strip_closing_vertex
from econ_viz.models import CobbDouglas


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class _GC:
    def __init__(
        self,
        *,
        rgb=(0.0, 0.0, 0.0),
        linewidth=1.0,
        alpha=1.0,
        dashes=None,
    ):
        self._rgb = rgb
        self._linewidth = linewidth
        self._alpha = alpha
        self._dashes = dashes

    def get_rgb(self):
        return self._rgb

    def get_linewidth(self):
        return self._linewidth

    def get_alpha(self):
        return self._alpha

    def get_dashes(self):
        return 0, self._dashes


def test_non_filled_path_strips_auto_appended_closing_vertex():
    poly = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])

    stripped = strip_closing_vertex(poly, filled=False)

    assert stripped.tolist() == [[0.0, 0.0], [1.0, 2.0]]


def test_filled_path_keeps_closing_vertex():
    poly = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])

    stripped = strip_closing_vertex(poly, filled=True)

    assert stripped.tolist() == poly.tolist()


def test_path_to_polylines_handles_open_two_point_spine_path():
    path = Path(np.array([[0.0, 0.0], [2.0, 0.0]]))

    polylines = path_to_polylines(path, IdentityTransform(), filled=False)

    assert len(polylines) == 1
    assert polylines[0].tolist() == [[0.0, 0.0], [2.0, 0.0]]


def test_renderer_emits_open_draw_path_without_return_segment():
    renderer = TikzRenderer(10, 10, scale=1.0)
    path = Path(np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]))

    renderer.draw_path(_GC(), path, IdentityTransform(), rgbFace=None)

    command = renderer.commands[-1]
    assert command.startswith(r"\draw")
    assert command.count("(0.0000,0.0000)") == 1


def test_renderer_emits_filldraw_with_registered_fill_color():
    renderer = TikzRenderer(10, 10, scale=1.0)
    path = Path(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))

    renderer.draw_path(
        _GC(rgb=(1.0, 0.0, 0.0), linewidth=2.5),
        path,
        IdentityTransform(),
        rgbFace=(0.0, 1.0, 0.0, 0.4),
    )
    document = assemble(renderer, standalone=False)

    assert r"\filldraw" in document
    assert "line width=2.50pt" in document
    assert "fill=" in document
    assert "fill opacity=0.400" in document
    assert document.count(r"\definecolor") == 2


def test_dash_spec_uses_tikz_named_styles_when_possible():
    assert dash_spec([4.0, 4.0]) == "dashed"
    assert dash_spec([1.0, 3.0]) == "dotted"
    assert dash_spec([5.0, 2.0, 1.0, 2.0]).startswith("dash pattern=")


def test_renderer_sanitizes_unicode_text_and_wraps_math_mode():
    renderer = TikzRenderer(10, 10, scale=1.0)

    renderer.draw_text(
        _GC(),
        1.0,
        2.0,
        "x−1",
        FontProperties(size=10),
        0.0,
        ismath=True,
    )

    command = renderer.commands[-1]
    assert "−" not in command
    assert "{$x-1$}" in command


def test_renderer_does_not_double_wrap_existing_math_text():
    renderer = TikzRenderer(10, 10, scale=1.0)

    renderer.draw_text(
        _GC(),
        1.0,
        2.0,
        "$x$",
        FontProperties(size=10),
        0.0,
        ismath=True,
    )

    assert "{$x$}" in renderer.commands[-1]
    assert "{$$x$$}" not in renderer.commands[-1]


def test_canvas_save_tex_writes_standalone_tikz(tmp_path):
    out = tmp_path / "fig.tex"

    Canvas().add_utility(CobbDouglas(), levels=2).save(str(out))

    text = out.read_text(encoding="utf-8")
    assert r"\documentclass[border=8pt]{standalone}" in text
    assert r"\usepackage{tikz}" in text
    assert r"\begin{tikzpicture}" in text
    assert r"\draw" in text
    assert "−" not in text


def test_canvas_save_tex_includes_visible_axis_spines(tmp_path):
    out = tmp_path / "axis.tex"

    Canvas(x_max=20, y_max=15).save(str(out))

    text = out.read_text(encoding="utf-8")
    assert "(0.9375,0.8250) -- (6.7500,0.8250);" in text
    assert "(0.9375,0.8250) -- (0.9375,6.6000);" in text


def test_save_figure_tex_passes_tikz_options(monkeypatch, tmp_path):
    calls = {}

    def fake_save_tikz(fig, path, *, scale, standalone):
        calls["fig"] = fig
        calls["path"] = path
        calls["scale"] = scale
        calls["standalone"] = standalone

    monkeypatch.setattr("econ_viz.io.backend_tikz.save_tikz", fake_save_tikz)
    fig = object()

    save_figure(
        fig,
        path=str(tmp_path / "fig.tex"),
        dpi=100,
        tikz_scale=0.4,
        tikz_standalone=False,
    )

    assert calls == {
        "fig": fig,
        "path": str(tmp_path / "fig.tex"),
        "scale": 0.4,
        "standalone": False,
    }


def test_default_scale_maps_six_inch_figure_to_seven_point_five_cm():
    renderer = TikzRenderer(600, 400, dpi=100)

    assert renderer.scale == pytest.approx(0.0125)
    assert renderer.width * renderer.scale == pytest.approx(7.5)

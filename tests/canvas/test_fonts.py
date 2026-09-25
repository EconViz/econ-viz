"""Tests for per-canvas font configuration."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from econ_viz import Canvas, Figure, Layout, levels, solve
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CobbDouglas


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _full_canvas(**kwargs) -> Canvas:
    model = CobbDouglas(alpha=0.5, beta=0.5)
    eq = solve(model, px=2.0, py=3.0, income=30.0)
    cvs = Canvas(x_max=20, y_max=15, title="Title", **kwargs)
    cvs.add_utility(model, levels=levels.around(eq.utility, n=3), label="IC", show_ic_labels=True)
    cvs.add_budget(2.0, 3.0, 30.0, label="BC")
    cvs.add_equilibrium(eq)
    cvs.add_point(12.0, 2.0, label="A")
    cvs.show_legend()
    return cvs


def _visible_texts(fig) -> list[Text]:
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def test_default_canvas_keeps_matplotlib_font():
    cvs = _full_canvas()
    cvs.fig.canvas.draw()

    assert cvs.font is None
    default = list(matplotlib.rcParams["font.family"])
    assert all(t.get_fontfamily() == default for t in _visible_texts(cvs.fig))


def test_custom_font_applies_to_every_text_element():
    cvs = _full_canvas(font="DejaVu Serif")
    cvs.fig.canvas.draw()

    texts = _visible_texts(cvs.fig)
    assert {"Title", "IC", "$BC$", "$0$", "$A$"} <= {t.get_text() for t in texts}
    assert all(t.get_fontfamily() == ["DejaVu Serif"] for t in texts)


def test_font_applies_to_layers_added_after_construction():
    cvs = Canvas(font="serif")
    cvs.add_point(1.0, 1.0, label="late")
    cvs.fig.canvas.draw()

    late = next(t for t in cvs.ax.texts if "late" in t.get_text())
    assert late.get_fontfamily() == ["serif"]


def test_font_does_not_change_global_rcparams():
    before = list(matplotlib.rcParams["font.family"])
    Canvas(font="DejaVu Serif").fig.canvas.draw()
    assert list(matplotlib.rcParams["font.family"]) == before


def test_font_accepts_fallback_list():
    cvs = Canvas(font=["No Such Font EconViz", "DejaVu Serif"])
    assert cvs.font == ("No Such Font EconViz", "DejaVu Serif")


def test_missing_font_raises_clear_error():
    with pytest.raises(InvalidParameterError, match="No Such Font EconViz"):
        Canvas(font="No Such Font EconViz")


def test_empty_font_is_rejected():
    with pytest.raises(InvalidParameterError):
        Canvas(font=[])


def test_figure_propagates_font_to_all_panels():
    fig = Figure(Layout.SIDE_BY_SIDE, title="Suptitle", font="DejaVu Serif")
    for panel in fig.canvases:
        panel.add_point(1.0, 1.0, label="P")
    fig.fig.canvas.draw()

    assert all(panel.font == ("DejaVu Serif",) for panel in fig.canvases)
    texts = _visible_texts(fig.fig)
    assert "Suptitle" in {t.get_text() for t in texts}
    assert all(t.get_fontfamily() == ["DejaVu Serif"] for t in texts)


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg"])
def test_custom_font_exports(tmp_path, suffix):
    path = tmp_path / f"font{suffix}"
    _full_canvas(font="DejaVu Serif").save(str(path))
    assert path.stat().st_size > 0


def test_tikz_maps_generic_serif_family(tmp_path):
    path = tmp_path / "serif.tex"
    Canvas(font="serif").save(str(path))
    tex = path.read_text()
    assert r"\rmfamily" in tex
    assert r"\sffamily" not in tex


def test_tikz_default_output_is_unchanged(tmp_path):
    path = tmp_path / "default.tex"
    Canvas().save(str(path))
    tex = path.read_text()
    assert r"\sffamily" in tex
    assert r"\rmfamily" not in tex


def test_math_font_applies_to_math_text_only():
    cvs = _full_canvas(font="DejaVu Serif", math_font="stix")
    cvs.fig.canvas.draw()

    texts = _visible_texts(cvs.fig)
    assert cvs.math_font == "stix"
    assert all(t.get_math_fontfamily() == "stix" for t in texts)


def test_default_math_font_is_unchanged():
    cvs = _full_canvas(font="DejaVu Serif")
    cvs.fig.canvas.draw()

    default = matplotlib.rcParams["mathtext.fontset"]
    assert all(t.get_math_fontfamily() == default for t in _visible_texts(cvs.fig))


def test_invalid_math_font_is_rejected():
    with pytest.raises(InvalidParameterError, match="stix"):
        Canvas(math_font="comic")


def test_figure_propagates_math_font():
    fig = Figure(Layout.SIDE_BY_SIDE, math_font="cm")
    fig.fig.canvas.draw()

    assert all(panel.math_font == "cm" for panel in fig.canvases)
    assert all(t.get_math_fontfamily() == "cm" for t in _visible_texts(fig.fig))

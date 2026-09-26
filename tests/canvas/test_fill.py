"""Tests for Fill: colour and opacity of the shaded budget set."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from econ_viz import Canvas, Fill, Stroke
from econ_viz.exceptions import InvalidParameterError
from econ_viz.themes.theme import Theme


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _fill(cvs):
    regions = [a for a in cvs.ax.get_children() if getattr(a, "_ev_role", None) == "budget_fill"]
    assert len(regions) <= 1
    return regions[0] if regions else None


def _looks_like(region, color, alpha):
    assert to_hex(region.get_facecolor()[0]) == to_hex(color)
    assert region.get_alpha() == pytest.approx(alpha)


class TestFillValue:
    def test_defaults(self):
        assert (Fill().color, Fill().alpha) == (None, None)

    @pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": 1.5}, {"color": "not-a-colour"}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Fill(**kwargs)

    def test_merged_over_keeps_unset_fields(self):
        merged = Fill(alpha=0.5).merged_over(Fill(color="red", alpha=0.1))
        assert (merged.color, merged.alpha) == ("red", 0.5)


class TestBudgetFill:
    def test_no_fill_by_default(self):
        assert _fill(Canvas().add_budget(2, 3, 30)) is None
        assert _fill(Canvas().add_budget(2, 3, 30, fill=False)) is None

    def test_fill_true_matches_theme_default(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=True)
        t = cvs.theme
        assert t.budget_fill == Fill(alpha=t.budget_fill_alpha)
        _looks_like(_fill(cvs), t.budget_color, t.budget_fill_alpha)

    def test_fill_colour_independent_of_line(self):
        cvs = Canvas().add_budget(2, 3, 30, color="black", fill=Fill(color="lightgrey", alpha=0.3))
        _looks_like(_fill(cvs), "lightgrey", 0.3)
        line = [a for a in cvs.ax.lines if getattr(a, "_ev_role", None) == "budget"][0]
        assert to_hex(line.get_color()) == to_hex("black")

    def test_fill_without_colour_follows_line(self):
        cvs = Canvas().add_budget(2, 3, 30, color="#123456", fill=Fill(alpha=0.2))
        _looks_like(_fill(cvs), "#123456", 0.2)

    def test_fill_without_colour_follows_stroke(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=True, stroke=Stroke(color="#654321"))
        _looks_like(_fill(cvs), "#654321", cvs.theme.budget_fill_alpha)

    def test_fill_alpha_still_works(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=True, fill_alpha=0.4)
        assert _fill(cvs).get_alpha() == pytest.approx(0.4)
        cvs = Canvas().add_budget(2, 3, 30, fill=Fill(color="green"), fill_alpha=0.4)
        _looks_like(_fill(cvs), "green", 0.4)

    def test_fill_alpha_loses_to_fill(self):
        cvs = Canvas().add_budget(2, 3, 30, fill=Fill(alpha=0.6), fill_alpha=0.4)
        assert _fill(cvs).get_alpha() == pytest.approx(0.6)

    def test_theme_default(self):
        class Grey(Theme):
            @property
            def budget_fill(self):
                return Fill(color="#cccccc", alpha=0.5)

        cvs = Canvas(theme=Grey(name="grey")).add_budget(2, 3, 30, fill=True)
        _looks_like(_fill(cvs), "#cccccc", 0.5)

    def test_fill_reaches_tikz(self, tmp_path):
        path = tmp_path / "f.tex"
        Canvas().add_budget(2, 3, 30, fill=Fill(color="#00ff00", alpha=0.25)).save(str(path))
        text = path.read_text()
        assert "fill opacity=0.250" in text


def test_fill_is_exported():
    import econ_viz

    assert econ_viz.Fill is Fill

"""Each line-styling shorthand parameter draws the same as the matching Stroke."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import Collection
from matplotlib.colors import to_hex
from matplotlib.text import Annotation

from econ_viz import Canvas, LinearBudget, PricePath, Stroke
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import decompose_price_effect

MODEL = CobbDouglas(alpha=0.5, beta=0.5)
DEC = decompose_price_effect(MODEL, px=(2.0, 4.0), py=3.0, income=30.0)
PATH = PricePath(MODEL, budget=LinearBudget(px=2.0, py=2.0, income=40.0), price="px",
                 price_range=(0.8, 6.0), n=5)
COLOR, WIDTH = "#123456", 2.2


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _look(ax, role):
    """Colour, width, and dash pattern of every artist with *role*."""
    looks = []
    for artist in ax.get_children():
        if getattr(artist, "_ev_role", None) != role:
            continue
        if isinstance(artist, Annotation):
            artist = artist.arrow_patch
            if artist is None:
                continue
            looks.append((to_hex(artist.get_edgecolor()), artist.get_linewidth(), str(artist.get_linestyle())))
        elif isinstance(artist, Collection):
            looks.append((to_hex(artist.get_edgecolor()[0]), artist.get_linewidth()[0], str(artist.get_linestyle()[0])))
        else:
            looks.append((to_hex(artist.get_color()), artist.get_linewidth(), str(artist.get_linestyle())))
    assert looks, f"nothing drawn with role {role!r}"
    return looks


def _box():
    return EdgeworthBox(MODEL, MODEL, total_x=10.0, total_y=10.0).add_endowment(7.0, 3.0)


CASES = [
    # (role, draw with shorthand, draw with Stroke)
    ("curve",
     lambda: Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5], color=COLOR, linewidth=WIDTH),
     lambda: Canvas(x_max=20, y_max=15).add_utility(MODEL, levels=[3, 5], stroke=Stroke(color=COLOR, width=WIDTH))),
    ("budget",
     lambda: Canvas().add_budget(2, 3, 30, color=COLOR, linewidth=WIDTH, linestyle="--"),
     lambda: Canvas().add_budget(2, 3, 30, stroke=Stroke(color=COLOR, width=WIDTH, style="--"))),
    ("ray",
     lambda: Canvas().add_ray(0.5, color=COLOR, linewidth=WIDTH),
     lambda: Canvas().add_ray(0.5, stroke=Stroke(color=COLOR, width=WIDTH))),
    ("path",
     lambda: Canvas(x_max=30, y_max=25).add_path(PATH, color=COLOR, linewidth=WIDTH),
     lambda: Canvas(x_max=30, y_max=25).add_path(PATH, stroke=Stroke(color=COLOR, width=WIDTH))),
    ("original_budget",
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, original_budget_color=COLOR, original_budget_linewidth=WIDTH, original_budget_linestyle=":"),
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, original_budget_stroke=Stroke(color=COLOR, width=WIDTH, style=":"))),
    ("compensated_budget",
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, compensated_budget_color=COLOR, compensated_budget_linewidth=WIDTH,
         compensated_budget_linestyle="-"),
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, compensated_budget_stroke=Stroke(color=COLOR, width=WIDTH, style="-"))),
    ("final_budget",
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, final_budget_color=COLOR, final_budget_linewidth=WIDTH, final_budget_linestyle="--"),
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(
         DEC, final_budget_stroke=Stroke(color=COLOR, width=WIDTH, style="--"))),
    ("substitution",
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(DEC, substitution_color=COLOR, effect_arrow_linewidth=WIDTH),
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(DEC, substitution_stroke=Stroke(color=COLOR, width=WIDTH))),
    ("income",
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(DEC, income_color=COLOR, effect_arrow_linewidth=WIDTH),
     lambda: Canvas(x_max=20, y_max=15).add_decomposition(DEC, income_stroke=Stroke(color=COLOR, width=WIDTH))),
    ("curve_a",
     lambda: _box().add_indifference_curves(color_a=COLOR, linewidth=WIDTH, res=60),
     lambda: _box().add_indifference_curves(stroke_a=Stroke(color=COLOR, width=WIDTH), res=60)),
    ("curve_b",
     lambda: _box().add_endowment_indifference(color_b=COLOR, linewidth=WIDTH, res=60),
     lambda: _box().add_endowment_indifference(stroke_b=Stroke(color=COLOR, width=WIDTH), res=60)),
    ("curve_a",
     lambda: _box().add_equilibrium_indifference(px=1.0, py=1.0, color_a=COLOR, linewidth=WIDTH, res=60),
     lambda: _box().add_equilibrium_indifference(px=1.0, py=1.0, stroke_a=Stroke(color=COLOR, width=WIDTH), res=60)),
    ("curve_b",
     lambda: _box().add_indifference_curves_from_equilibrium(px=1.0, py=1.0, color_b=COLOR, linewidth=WIDTH, res=60),
     lambda: _box().add_indifference_curves_from_equilibrium(
         px=1.0, py=1.0, stroke_b=Stroke(color=COLOR, width=WIDTH), res=60)),
    ("contract",
     lambda: _box().add_contract_curve(color=COLOR, linewidth=WIDTH, linestyle=":"),
     lambda: _box().add_contract_curve(stroke=Stroke(color=COLOR, width=WIDTH, style=":"))),
    ("core",
     lambda: _box().add_contract_curve().add_core(color=COLOR, linewidth=WIDTH),
     lambda: _box().add_contract_curve().add_core(stroke=Stroke(color=COLOR, width=WIDTH))),
    ("price",
     lambda: _box().add_price_line(px=1.0, py=1.0, color=COLOR, linewidth=WIDTH, linestyle="-."),
     lambda: _box().add_price_line(px=1.0, py=1.0, stroke=Stroke(color=COLOR, width=WIDTH, style="-."))),
]


@pytest.mark.parametrize("role, shorthand, stroke", CASES, ids=[f"{i}-{c[0]}" for i, c in enumerate(CASES)])
def test_shorthand_matches_stroke(role, shorthand, stroke):
    assert _look(shorthand().ax, role) == _look(stroke().ax, role)


@pytest.mark.parametrize("role, shorthand, stroke", CASES[:2], ids=["curve", "budget"])
def test_shorthand_emits_no_warning(recwarn, role, shorthand, stroke):
    shorthand()
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

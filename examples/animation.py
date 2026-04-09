"""Animated GIF examples for econ_viz v1.4.0.

Demonstrates the Animator class (factory pattern): each GIF is generated
by sweeping one economic parameter across a range and saving the result
with a single ``.save()`` call.  No ffmpeg required — only Pillow.

Run from the project root::

    python examples/animation.py

Output is written to ``examples/output/animation/``.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from econ_viz import Canvas
from econ_viz.models import CobbDouglas
from econ_viz.optimizer import solve
from econ_viz.animation import Animator

OUTPUT_DIR = "examples/output/animation"


# ──────────────────────────────────────────────────────────────────────
# GIF 1 — Price sweep (P₁ falling: budget rotates outward)
#
#   As P₁ decreases from 8 → 1 the budget constraint pivots around the
#   Y-intercept (income / P₂), expanding the affordable set.
#   The optimal bundle traces the Price-Consumption Curve (PCC).
# ──────────────────────────────────────────────────────────────────────

MODEL   = CobbDouglas(alpha=0.5, beta=0.5)
P2      = 2.0
INCOME  = 20.0

def draw_price_sweep(p1: float) -> Canvas:
    """Return a Canvas showing budget + indifference curves for a given P₁."""
    c = Canvas(
        x_max=14,
        y_max=12,
        x_label="X_1",
        y_label="X_2",
        title=rf"Price sweep — $P_1 = {p1:.2f}$",
        dpi=120,
    )
    c.add_utility(MODEL, levels=6)
    c.add_budget(px=p1, py=P2, income=INCOME, fill=True)

    eq = solve(MODEL, px=p1, py=P2, income=INCOME)
    c.add_equilibrium(eq, drop_dashes=True)

    return c


price_frames = np.linspace(8.0, 1.0, 50)   # P₁ falls: budget pivots out

print("Rendering GIF 1/2 — price sweep …")
Animator(draw_price_sweep, frames=price_frames).save(
    f"{OUTPUT_DIR}/price_sweep.gif",
    fps=12,
    dpi=120,
)
print(f"  → saved to {OUTPUT_DIR}/price_sweep.gif")


# ──────────────────────────────────────────────────────────────────────
# GIF 2 — Income sweep (Engel expansion path)
#
#   As income rises from 5 → 40 the budget constraint shifts outward
#   in parallel.  For a Cobb-Douglas utility the optimal bundle moves
#   along a straight ray through the origin (income-consumption curve).
# ──────────────────────────────────────────────────────────────────────

P1_FIXED = 3.0
P2_FIXED = 2.0

def draw_income_sweep(income: float) -> Canvas:
    """Return a Canvas showing budget + equilibrium for a given income level."""
    c = Canvas(
        x_max=14,
        y_max=12,
        x_label="X_1",
        y_label="X_2",
        title=rf"Income expansion — $I = {income:.1f}$",
        dpi=120,
    )
    c.add_utility(MODEL, levels=6)
    c.add_budget(px=P1_FIXED, py=P2_FIXED, income=income, fill=True)

    eq = solve(MODEL, px=P1_FIXED, py=P2_FIXED, income=income)
    c.add_equilibrium(eq, drop_dashes=True, show_ray=True)

    return c


income_frames = np.linspace(5.0, 40.0, 50)

print("Rendering GIF 2/2 — income expansion …")
Animator(draw_income_sweep, frames=income_frames).save(
    f"{OUTPUT_DIR}/income_expansion.gif",
    fps=12,
    dpi=120,
)
print(f"  → saved to {OUTPUT_DIR}/income_expansion.gif")

print("\nDone. Both GIFs written to", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────
# Jupyter interactive usage (reference only — run inside a notebook)
# ──────────────────────────────────────────────────────────────────────

JUPYTER_EXAMPLE = """
# ── Paste into a Jupyter notebook cell ──────────────────────────────
from econ_viz.interactive import WidgetViewer
from econ_viz import Canvas
from econ_viz.models import CobbDouglas

MODEL = CobbDouglas(alpha=0.5, beta=0.5)

def draw(p1: float, income: float) -> Canvas:
    c = Canvas(x_max=14, y_max=12, x_label="X_1", y_label="X_2")
    c.add_utility(MODEL, levels=6)
    c.add_budget(px=p1, py=2.0, income=income, fill=True)
    from econ_viz.optimizer import solve
    eq = solve(MODEL, px=p1, py=2.0, income=income)
    c.add_equilibrium(eq, drop_dashes=True, show_ray=True)
    return c

WidgetViewer(draw, p1=(1.0, 8.0, 0.5), income=(5.0, 40.0, 5.0)).show()
# ────────────────────────────────────────────────────────────────────
"""

"""Restyle every line with Stroke: theme defaults on the left, custom strokes on the right."""

from pathlib import Path

import matplotlib.pyplot as plt

from econ_viz import ArrowStyle, Canvas, Stroke, levels, solve
from econ_viz.models import CobbDouglas

OUTPUT_DIR = Path("examples/output/customization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = CobbDouglas(alpha=0.5, beta=0.5)
eq = solve(model, px=2.0, py=3.0, income=30.0)
lvls = levels.around(eq.utility, n=3)

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))

Canvas(x_max=20, y_max=15, title="Theme defaults", fig=fig, ax=left).add_utility(model, levels=lvls).add_budget(
    2.0, 3.0, 30.0
).add_equilibrium(eq, show_ray=True)

Canvas(
    x_max=20,
    y_max=15,
    title="Custom strokes",
    fig=fig,
    ax=right,
    axis_stroke=Stroke(width=1.4, arrow=ArrowStyle.SIMPLE),
).add_utility(model, levels=lvls, stroke=Stroke(width=1.0, style="dashed", arrow=ArrowStyle.TRIANGLE)).add_budget(
    2.0, 3.0, 30.0, stroke=Stroke(width=3.0, color="black")
).add_equilibrium(
    eq,
    show_ray=True,
    drop_stroke=Stroke(width=1.2, style="dashdot"),
    ray_stroke=Stroke(width=1.2, style="solid", arrow=ArrowStyle.WEDGE),
)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "stroke_customization.png", dpi=200, transparent=True)

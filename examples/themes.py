"""Render the same diagram under all built-in themes."""

from econ_viz import Canvas, levels, solve, themes
from econ_viz.models import CobbDouglas

OUTPUT_DIR = "examples/output"

model = CobbDouglas(0.5, 0.5)
px, py, income = 2.0, 3.0, 30.0
eq = solve(model, px, py, income)
lvls = levels.around(eq.utility, n=5)

for theme in [themes.default, themes.nord]:
    cvs = Canvas(x_max=20, y_max=15, x_label="Pizza", y_label="Cola", theme=theme)
    cvs.add_utility(model, levels=lvls, show_rays=True)
    cvs.add_budget(px, py, income, fill=True)
    cvs.add_equilibrium(eq, show_ray=True)
    cvs.save(f"{OUTPUT_DIR}/theme_{theme.name}.png")

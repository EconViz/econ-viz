"""Plot indifference curves for all four utility models."""

from econ_viz.canvas.base import Canvas
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES

OUTPUT_DIR = "examples/output"

models = [
    ("cobb_douglas", CobbDouglas(0.5, 0.5)),
    ("leontief", Leontief(2, 3)),
    ("perfect_substitutes", PerfectSubstitutes(1, 2)),
    ("ces", CES(0.5, 0.5, 0.5)),
]

for name, model in models:
    cvs = Canvas(x_max=20, y_max=15, x_label="Pizza", y_label="Cola")
    cvs.add_utility(model, levels=5, show_rays=True, show_kinks=True)
    cvs.save(f"{OUTPUT_DIR}/{name}.png")

"""Draw the same consumer equilibrium with the default and a serif font."""

from econ_viz import Figure, Layout, levels, solve
from econ_viz.models import CobbDouglas

model = CobbDouglas(alpha=0.5, beta=0.5)
eq = solve(model, px=2.0, py=3.0, income=30.0)

# Fallback list: the first installed family is used.
fig = Figure(
    Layout.SIDE_BY_SIDE,
    x_max=20,
    y_max=15,
    title="Consumer equilibrium",
    font=["Times New Roman", "DejaVu Serif", "serif"],
    math_font="stix",
)
for panel, title in zip(fig.canvases, ["Utility map", "With budget"]):
    panel.ax.set_title(title)
    panel.add_utility(model, levels=levels.around(eq.utility, n=3), label="IC")
fig[1].add_budget(2.0, 3.0, 30.0, fill=True, label="BC").add_equilibrium(eq).show_legend()

fig.save("custom_font.png")

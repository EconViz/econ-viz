"""Compare axis-label positions, arrowhead styles, and axis line styles."""

import matplotlib.pyplot as plt

from econ_viz import ArrowStyle, Canvas, LabelPosition, LineStyle


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
settings = [
    (ArrowStyle.TRIANGLE, LineStyle.SOLID, LabelPosition.RIGHT, LabelPosition.TOP),
    (ArrowStyle.SIMPLE, LineStyle.DASHED, LabelPosition.TOP, LabelPosition.LEFT),
    (ArrowStyle.FANCY, LineStyle.DOTTED, LabelPosition.BOTTOM, LabelPosition.RIGHT),
    (ArrowStyle.WEDGE, LineStyle.DASHDOT, LabelPosition.RIGHT, LabelPosition.RIGHT),
]

for ax, (arrow, line, x_position, y_position) in zip(axes.flat, settings):
    Canvas(
        title=f"{arrow.name.title()} · {line.name.title()}",
        x_label_pos=x_position,
        y_label_pos=y_position,
        x_arrow_style=arrow,
        y_arrow_style=arrow,
        x_line_style=line,
        y_line_style=line,
        fig=fig,
        ax=ax,
    )

fig.tight_layout()
fig.savefig("axis_customization.png", dpi=200, transparent=True)

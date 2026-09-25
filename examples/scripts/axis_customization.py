"""Compare the supported axis-label positions and arrowhead styles."""

import matplotlib.pyplot as plt

from econ_viz import ArrowStyle, Canvas, LabelPosition


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
settings = [
    (ArrowStyle.TRIANGLE, LabelPosition.RIGHT, LabelPosition.TOP),
    (ArrowStyle.SIMPLE, LabelPosition.TOP, LabelPosition.LEFT),
    (ArrowStyle.FANCY, LabelPosition.BOTTOM, LabelPosition.RIGHT),
    (ArrowStyle.WEDGE, LabelPosition.RIGHT, LabelPosition.RIGHT),
]

for ax, (style, x_position, y_position) in zip(axes.flat, settings):
    Canvas(
        title=style.name.title(),
        x_label_pos=x_position,
        y_label_pos=y_position,
        x_arrow_style=style,
        y_arrow_style=style,
        fig=fig,
        ax=ax,
    )

fig.tight_layout()
fig.savefig("axis_customization.png", dpi=200, transparent=True)

"""Frame styling (background, limits, spines) for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from utility_viz.core.diagrams.consumer.edgeworth.style.labels import add_labels
from utility_viz.enums import LineStyle

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.base import BoxBase


def apply_base_style(box: BoxBase) -> None:
    """Apply background, limits, spines, and fixed-place labels to *box*."""
    t = box.theme
    if t.background_color is not None:
        box.fig.patch.set_facecolor(t.background_color)
        box.fig.patch.set_alpha(1.0)
        box.ax.patch.set_facecolor(t.background_color)
        box.ax.patch.set_alpha(1.0)
    box.ax.set_xlim(0.0, box.total_x)
    box.ax.set_ylim(0.0, box.total_y)
    box.ax.set_xticks([])
    box.ax.set_yticks([])
    _style_spines(box)
    add_labels(box)


def _style_spines(box: BoxBase) -> None:
    for side in ("top", "right", "bottom", "left"):
        stroke = box.x_side_stroke if side in ("top", "bottom") else box.y_side_stroke
        box.ax.spines[side].set_visible(True)
        box.ax.spines[side].set_color(stroke.color)
        box.ax.spines[side].set_linewidth(stroke.width)
        box.ax.spines[side].set_linestyle(cast(LineStyle, stroke.style).value)
        box.ax.spines[side].set_alpha(stroke.opacity)

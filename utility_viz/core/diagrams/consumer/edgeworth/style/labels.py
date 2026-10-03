"""Fixed-place text (axis names, origins, title) for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utility_viz.core.styles.label import Label

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.base import BoxBase


def styled_text(box: BoxBase, text, style: Label):
    """Apply a :class:`Label`'s colour, size, visibility and opacity to *text*."""
    text.set_color(style.color or box.theme.label_color)
    if style.fontsize is not None:
        text.set_fontsize(style.fontsize)
    text.set_visible(style.visible is not False)
    text.set_alpha(style.opacity)
    return text


def add_labels(box: BoxBase) -> None:
    """Add ``x_A``/``y_A``, ``O_A``/``O_B``, ``x_B``/``y_B`` and the title."""
    ax, tx, ty = box.ax, box.total_x, box.total_y
    styled_text(box, ax.set_xlabel(rf"${box.x_label}_A$"), box.x_label_style)
    styled_text(box, ax.set_ylabel(rf"${box.y_label}_A$"), box.y_label_style)
    styled_text(box, ax.text(0.0, 0.0, r"$O_A$", ha="right", va="top"), box.origin_style)
    styled_text(box, ax.text(tx, ty, r"$O_B$", ha="left", va="bottom"), box.origin_style)
    styled_text(
        box,
        ax.text(tx * 0.98, ty * -0.06, rf"${box.x_label}_B$", ha="right", va="top"),
        box.x_label_style,
    )
    styled_text(
        box,
        ax.text(tx * -0.04, ty * 0.98, rf"${box.y_label}_B$", ha="right", va="top", rotation=90),
        box.y_label_style,
    )
    if box.title:
        styled_text(box, ax.set_title(box.title), box.title_style)

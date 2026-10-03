"""Base styling (spines, axis labels, origin labels, title) for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from utility_viz.core.styles.label import Label
from utility_viz.enums import LineStyle

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth import EdgeworthBox


def apply_base_style(box: EdgeworthBox) -> None:
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
    _add_labels(box)


def _style_spines(box: EdgeworthBox) -> None:
    for side in ("top", "right", "bottom", "left"):
        stroke = box.x_side_stroke if side in ("top", "bottom") else box.y_side_stroke
        box.ax.spines[side].set_visible(True)
        box.ax.spines[side].set_color(stroke.color)
        box.ax.spines[side].set_linewidth(stroke.width)
        box.ax.spines[side].set_linestyle(cast(LineStyle, stroke.style).value)
        box.ax.spines[side].set_alpha(stroke.opacity)


def _styled_text(box: EdgeworthBox, text, style: Label):
    text.set_color(style.color or box.theme.label_color)
    if style.fontsize is not None:
        text.set_fontsize(style.fontsize)
    text.set_visible(style.visible is not False)
    text.set_alpha(style.opacity)
    return text


def _add_labels(box: EdgeworthBox) -> None:
    ax, tx, ty = box.ax, box.total_x, box.total_y
    _styled_text(box, ax.set_xlabel(rf"${box.x_label}_A$"), box.x_label_style)
    _styled_text(box, ax.set_ylabel(rf"${box.y_label}_A$"), box.y_label_style)
    _styled_text(box, ax.text(0.0, 0.0, r"$O_A$", ha="right", va="top"), box.origin_style)
    _styled_text(box, ax.text(tx, ty, r"$O_B$", ha="left", va="bottom"), box.origin_style)
    _styled_text(
        box,
        ax.text(tx * 0.98, ty * -0.06, rf"${box.x_label}_B$", ha="right", va="top"),
        box.x_label_style,
    )
    _styled_text(
        box,
        ax.text(tx * -0.04, ty * 0.98, rf"${box.y_label}_B$", ha="right", va="top", rotation=90),
        box.y_label_style,
    )
    if box.title:
        _styled_text(box, ax.set_title(box.title), box.title_style)

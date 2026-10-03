"""Attribute declarations shared by the :class:`Canvas` mixins and style helpers."""

from __future__ import annotations

from typing import Any

from utility_viz.core.styles.label import Label
from utility_viz.core.styles.stroke import Stroke
from utility_viz.core.themes.theme import Theme
from utility_viz.enums import LabelPosition


class CanvasState:
    """Typed view of the attributes ``Canvas.__init__`` sets; no behaviour."""

    fig: Any
    ax: Any
    theme: Theme
    x_max: float
    y_max: float
    dpi: int
    title: str | None
    title_style: Label
    x_label: str
    y_label: str
    x_label_pos: LabelPosition
    y_label_pos: LabelPosition
    x_label_style: Label
    y_label_style: Label
    origin_text: str
    origin_style: Label
    x_axis_stroke: Stroke
    y_axis_stroke: Stroke
    _owns_figure: bool
    _legend_handles: list

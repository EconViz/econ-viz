"""Axis: one axis's label, label position, and stroke."""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import LabelPosition
from .label import to_position
from .stroke import Stroke


@dataclass(frozen=True)
class Axis:
    """Settings for one axis. Fields left as ``None`` keep the default.

    Parameters
    ----------
    label : str, optional
        Axis label, rendered in math mode.
    label_position : LabelPosition or str, optional
        Where the label sits around the arrowhead: ``"top"``, ``"right"``, or
        ``"bottom"`` for the x-axis; ``"left"``, ``"top"``, or ``"right"`` for
        the y-axis.
    stroke : Stroke, optional
        Width, line style, colour, and arrowhead of the axis line.
    """

    label: str | None = None
    label_position: LabelPosition | str | None = None
    stroke: Stroke | None = None

    def __post_init__(self) -> None:
        if self.label_position is not None:
            object.__setattr__(self, "label_position", to_position(self.label_position, what="Axis label_position"))

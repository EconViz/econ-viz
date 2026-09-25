"""Effect: colour, position, and label of one decomposition effect arrow."""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import LabelPosition
from ..exceptions import InvalidParameterError


@dataclass(frozen=True)
class Effect:
    """How to show the substitution or income effect in a decomposition.

    Parameters
    ----------
    color : str, optional
        Colour of the effect arrow, its range arrow below the x-axis, its
        label, and its legend entry.
    y : float, optional
        Height of the range arrow below the x-axis, as a fraction of the axes
        height (negative is below the axis). Used with ``show_x_projections``.
    label : str, optional
        Text drawn next to the arrow.
    label_position : LabelPosition or str
        Side of the arrow's middle where the text goes. For the horizontal
        range arrows below the x-axis, ``"left"`` / ``"right"`` place it past
        the arrow's left or right end. ``"above"`` and ``"below"`` are
        accepted as aliases for ``"top"`` and ``"bottom"``.
    label_offset : float
        Distance between the arrow and the text, in points.
    """

    color: str | None = None
    y: float | None = None
    label: str | None = None
    label_position: LabelPosition | str = LabelPosition.RIGHT
    label_offset: float = 4.0

    def __post_init__(self) -> None:
        value = {"above": "top", "below": "bottom"}.get(self.label_position, self.label_position)
        try:
            object.__setattr__(self, "label_position", LabelPosition(value))
        except ValueError:
            choices = ", ".join(p.value for p in LabelPosition)
            raise InvalidParameterError(
                f"invalid Effect label_position {self.label_position!r}; choose: {choices}"
            ) from None

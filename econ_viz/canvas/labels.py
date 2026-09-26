"""Placing text beside a point: offsets and alignment for each LabelPosition."""

from __future__ import annotations

from ..enums import LabelPosition

# Unit direction from the point to the text, then horizontal / vertical alignment.
LAYOUT = {
    LabelPosition.TOP: ((0, 1), "center", "bottom"),
    LabelPosition.BOTTOM: ((0, -1), "center", "top"),
    LabelPosition.LEFT: ((-1, 0), "right", "center"),
    LabelPosition.RIGHT: ((1, 0), "left", "center"),
    LabelPosition.TOP_RIGHT: ((1, 1), "left", "bottom"),
    LabelPosition.TOP_LEFT: ((-1, 1), "right", "bottom"),
    LabelPosition.BOTTOM_RIGHT: ((1, -1), "left", "top"),
    LabelPosition.BOTTOM_LEFT: ((-1, -1), "right", "top"),
}


def placement(position: LabelPosition, offset: float) -> tuple[tuple[float, float], str, str]:
    """Return ``(xytext in points, ha, va)`` for a label *offset* points away on *position*'s side.

    Corners move *offset* along each axis, matching the ``(5, 5)`` default of point labels.
    """
    (dx, dy), ha, va = LAYOUT[position]
    return (dx * offset, dy * offset), ha, va

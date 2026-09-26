"""Legend placement."""

from enum import Enum


class LegendPosition(str, Enum):
    """Where a legend goes: automatic, an inside corner, or outside the plot area."""

    AUTO = "auto"
    UPPER_RIGHT = "upper right"
    UPPER_LEFT = "upper left"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def inside(self) -> bool:
        return self in _INSIDE


_INSIDE = frozenset(
    {
        LegendPosition.UPPER_RIGHT,
        LegendPosition.UPPER_LEFT,
        LegendPosition.LOWER_LEFT,
        LegendPosition.LOWER_RIGHT,
    }
)

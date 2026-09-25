"""Enumerations for axis-label placement and arrowhead styling."""

from enum import Enum


class LabelPosition(str, Enum):
    """Cardinal positions used to place labels around an axis arrowhead."""

    TOP = "top"
    LEFT = "left"
    BOTTOM = "bottom"
    RIGHT = "right"


class ArrowStyle(str, Enum):
    """Matplotlib-compatible arrowhead styles for canvas axes."""

    SIMPLE = "->"
    TRIANGLE = "-|>"
    FANCY = "fancy"
    WEDGE = "wedge"

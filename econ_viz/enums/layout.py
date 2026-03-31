"""Enumeration of supported multi-panel figure layouts."""

from enum import Enum


class Layout(Enum):
    """Fixed-layout panel arrangements for :class:`econ_viz.Figure`."""

    SINGLE = "single"
    STACKED = "stacked"
    SIDE_BY_SIDE = "side_by_side"
    TOP_TWO_BOTTOM_ONE = "top_two_bottom_one"
    TOP_ONE_BOTTOM_TWO = "top_one_bottom_two"
    GRID_2X2 = "grid_2x2"
    GRID_3X3 = "grid_3x3"

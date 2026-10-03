"""
econ_viz.canvas — Plotting primitives and figure assembly.

This subpackage contains the core :class:`Canvas` class responsible for
axis styling, layer composition, and export, as well as the :class:`Layer`
helper that handles numerical evaluation of contour data.
"""

from econ_viz.canvas.base import Canvas
from econ_viz.canvas.figure import Figure
from econ_viz.canvas.layers import Layer

__all__ = ["Canvas", "Figure", "Layer"]

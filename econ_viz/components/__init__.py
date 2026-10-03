"""Reusable drawing components for economic diagrams."""

from econ_viz.canvas.stroke import tag
from econ_viz.components.budget import BudgetConstraint
from econ_viz.components.equilibrium import EquilibriumPoint
from econ_viz.components.indifference import IndifferenceCurves

__all__ = ["IndifferenceCurves", "BudgetConstraint", "EquilibriumPoint"]


def draw_ray(ax, slope, x_max, y_max, color, linewidth, linestyle="--"):
    """Draw a dashed ray from the origin, clipped to the visible area."""
    x_end = x_max
    y_end = slope * x_end
    if y_end > y_max:
        y_end = y_max
        x_end = y_end / slope
    (line,) = ax.plot([0, x_end], [0, y_end], color=color, linestyle=linestyle, linewidth=linewidth)
    tag(line, "ray")
    return line

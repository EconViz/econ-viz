"""Reusable drawing components for economic diagrams."""

from .indifference import IndifferenceCurves
from .budget import BudgetConstraint
from .equilibrium import EquilibriumPoint

__all__ = ["IndifferenceCurves", "BudgetConstraint", "EquilibriumPoint"]


def draw_ray(ax, slope, x_max, y_max, color, linewidth):
    """Draw a dashed ray from the origin, clipped to the visible area."""
    x_end = x_max
    y_end = slope * x_end
    if y_end > y_max:
        y_end = y_max
        x_end = y_end / slope
    ax.plot([0, x_end], [0, y_end],
            color=color, linestyle="--", linewidth=linewidth)

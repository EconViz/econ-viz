"""Indifference curve component."""

from __future__ import annotations

from ..enums import UtilityType
from ..logging import get_logger

logger = get_logger(__name__)


class IndifferenceCurves:
    """Renders a family of indifference curves for a utility function.

    Parameters
    ----------
    func : UtilityFunction
        Utility model conforming to the protocol.
    levels : int or list[float]
        Number of auto-spaced levels, or an explicit list of utility values.
    color, linewidth : str, float
        Curve appearance.
    show_rays : bool
        Draw kink-locus rays (only for KINKED utility types).
    ray_color, ray_linewidth : str, float
        Ray appearance.
    show_kinks : bool
        Draw markers at kink points (only for KINKED utility types).
    kink_color : str
        Kink marker colour.
    kink_radius : float
        Kink marker size factor.
    """

    def __init__(
        self,
        func,
        levels,
        color: str,
        linewidth: float,
        show_rays: bool = False,
        ray_color: str = "black",
        ray_linewidth: float = 0.8,
        show_kinks: bool = False,
        kink_color: str = "black",
        kink_radius: float = 1.0,
    ):
        self.func = func
        self.levels = levels
        self.color = color
        self.linewidth = linewidth
        self.show_rays = show_rays
        self.ray_color = ray_color
        self.ray_linewidth = ray_linewidth
        self.show_kinks = show_kinks
        self.kink_color = kink_color
        self.kink_radius = kink_radius

    def draw(self, ax, x_max: float, y_max: float, **kwargs) -> list[float]:
        """Draw curves onto *ax* and return the computed contour levels."""
        from ..canvas.layers import Layer
        from ..levels import percentile
        from . import draw_ray

        X, Y, Z = Layer.compute_contour(self.func, (0.1, x_max), (0.1, y_max))

        if isinstance(self.levels, int):
            computed = percentile(Z, n=self.levels)
        else:
            computed = list(self.levels)

        logger.debug("Drawing contours at levels: %s", computed)

        ax.contour(
            X, Y, Z, levels=computed,
            colors=self.color, linewidths=self.linewidth, **kwargs,
        )

        if self.show_rays and hasattr(self.func, "utility_type"):
            if self.func.utility_type is UtilityType.KINKED:
                for slope in self.func.ray_slopes():
                    draw_ray(ax, slope, x_max, y_max,
                             color=self.ray_color, linewidth=self.ray_linewidth)

        if self.show_kinks and hasattr(self.func, "utility_type"):
            if self.func.utility_type is UtilityType.KINKED:
                for x, y in self.func.kink_points(computed):
                    ax.plot(x, y, "o",
                            markersize=self.kink_radius * 4,
                            markerfacecolor=self.kink_color,
                            markeredgecolor=self.kink_color)

        return computed

"""Indifference curve component."""

from __future__ import annotations

import numpy as np

from ..contours import percentile_levels
from ..enums import UtilityType
from ..utils.logging import get_logger

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
        label: str | None = None,
        show_ic_labels: bool = False,
        ic_label_fmt: str = "{:.2g}",
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
        self.label = label
        self.show_ic_labels = show_ic_labels
        self.ic_label_fmt = ic_label_fmt

    def draw(self, ax, x_max: float, y_max: float, **kwargs) -> list[float]:
        """Draw curves onto *ax* and return the computed contour levels."""
        from ..canvas.layers import Layer
        from . import draw_ray

        res = int(kwargs.pop("res", 400))
        X, Y, Z = Layer.compute_contour(self.func, (0.1, x_max), (0.1, y_max), res=res)

        if isinstance(self.levels, int):
            computed = percentile_levels(Z, n=self.levels)
        else:
            computed = list(self.levels)

        logger.debug("Drawing contours at levels: %s", computed)

        cs = ax.contour(
            X, Y, Z, levels=computed,
            colors=self.color, linewidths=self.linewidth, **kwargs,
        )

        if self.label is not None:
            import matplotlib.lines as mlines
            self._proxy = mlines.Line2D(
                [], [], color=self.color, linewidth=self.linewidth, label=self.label
            )
        else:
            self._proxy = None

        if self.show_ic_labels:
            for level, segs in zip(computed, cs.allsegs):
                best_x, best_y = -1.0, None
                for seg in segs:
                    if len(seg) == 0:
                        continue
                    mask = (seg[:, 0] < x_max * 0.97) & (seg[:, 1] < y_max * 0.97)
                    seg = seg[mask]
                    if len(seg) == 0:
                        continue
                    idx = np.argmax(seg[:, 0])
                    if seg[idx, 0] > best_x:
                        best_x, best_y = seg[idx, 0], seg[idx, 1]
                if best_y is not None:
                    ax.text(
                        best_x + x_max * 0.01, best_y,
                        self.ic_label_fmt.format(level),
                        color=self.color, fontsize=9, va="center",
                        clip_on=True,
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

        if hasattr(self.func, "subsistence_lines"):
            sub_x, sub_y = self.func.subsistence_lines()
            style = dict(color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(x=sub_x, **style)
            ax.axhline(y=sub_y, **style)

        return computed

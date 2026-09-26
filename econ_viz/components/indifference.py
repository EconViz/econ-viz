"""Indifference curve component."""

from __future__ import annotations

import numpy as np

from ..canvas.stroke import tag, tag_attr
from ..constants.canvas import CONTOUR_DOMAIN_MIN
from ..contours import percentile_levels
from ..enums import LabelPosition, UtilityType
from ..themes.label import Label
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _label_angle(ax, segment: np.ndarray, index: int) -> float:
    """Return the local contour angle in display coordinates, kept upright."""
    if len(segment) < 2:
        return 0.0
    before = max(0, index - 1)
    after = min(len(segment) - 1, index + 1)
    start, end = ax.transData.transform([segment[before], segment[after]])
    angle = float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))
    return (angle + 90.0) % 180.0 - 90.0


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
    subsistence_color, subsistence_linewidth : str, float
        Appearance of Stone-Geary subsistence reference lines.
    highlight_level : float, optional
        Utility level to draw with full ``color``/``linewidth`` weight; every
        other level is drawn subdued, using *secondary_color* /
        *secondary_linewidth* / *secondary_opacity*. ``None`` draws every
        level with the same weight (unchanged default behaviour).
    secondary_color, secondary_linewidth, secondary_opacity : str, float, float
        Appearance of non-highlighted levels when *highlight_level* is set.
    label_style : str
        ``"numeric"`` (default) formats labels with *ic_label_fmt*;
        ``"ordinal"`` labels levels ``u_1, u_2, ...`` in ascending order.
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
        subsistence_color: str = "gray",
        subsistence_linewidth: float = 0.8,
        highlight_level: float | None = None,
        secondary_color: str = "gray",
        secondary_linewidth: float = 1.0,
        secondary_opacity: float = 0.45,
        label_style: str = "numeric",
    ):
        self.func = func
        self.levels = levels
        self.color = color
        self.linewidth = linewidth
        self.subsistence_color = subsistence_color
        self.subsistence_linewidth = subsistence_linewidth
        self.show_rays = show_rays
        self.ray_color = ray_color
        self.ray_linewidth = ray_linewidth
        self.show_kinks = show_kinks
        self.kink_color = kink_color
        self.kink_radius = kink_radius
        self.label = label
        self.show_ic_labels = show_ic_labels
        self.ic_label_fmt = ic_label_fmt
        self.highlight_level = highlight_level
        self.secondary_color = secondary_color
        self.secondary_linewidth = secondary_linewidth
        self.secondary_opacity = secondary_opacity
        self.label_style = label_style

    def draw(self, ax, x_max: float, y_max: float, **kwargs) -> list[float]:
        """Draw curves onto *ax* and return the computed contour levels."""
        from ..canvas.layers import Layer
        from . import draw_ray

        res = int(kwargs.pop("res", 400))
        X, Y, Z = Layer.compute_contour(self.func, (CONTOUR_DOMAIN_MIN, x_max), (CONTOUR_DOMAIN_MIN, y_max), res=res)

        computed = percentile_levels(Z, n=self.levels) if isinstance(self.levels, int) else list(self.levels)

        logger.debug("Drawing contours at levels: %s", computed)

        # Matplotlib dashes negative contour levels by default; utility levels are just levels.
        kwargs.setdefault("linestyles", "solid")

        focal_idx: int | None = None
        if self.highlight_level is not None and computed:
            focal_idx = int(np.argmin(np.abs(np.array(computed) - self.highlight_level)))

        focal_levels = computed if focal_idx is None else [computed[focal_idx]]
        secondary_levels = [] if focal_idx is None else [lv for i, lv in enumerate(computed) if i != focal_idx]

        cs = ax.contour(X, Y, Z, levels=focal_levels, colors=self.color, linewidths=self.linewidth, **kwargs)
        tag(cs, "curve")

        segs_by_level: dict[float, tuple] = dict(zip(focal_levels, cs.allsegs, strict=True))
        color_by_level: dict[float, str] = dict.fromkeys(focal_levels, self.color)

        if secondary_levels:
            cs2 = ax.contour(
                X,
                Y,
                Z,
                levels=secondary_levels,
                colors=self.secondary_color,
                linewidths=self.secondary_linewidth,
                alpha=self.secondary_opacity,
                **kwargs,
            )
            tag(cs2, "secondary_curve")
            segs_by_level.update(zip(secondary_levels, cs2.allsegs, strict=True))
            color_by_level.update(dict.fromkeys(secondary_levels, self.secondary_color))

        import matplotlib.lines as mlines

        self._proxy: mlines.Line2D | None = None
        if self.label is not None:
            self._proxy = mlines.Line2D([], [], color=self.color, linewidth=self.linewidth, label=self.label)
            tag(self._proxy, "curve")

        if self.show_ic_labels:
            for rank, level in enumerate(computed, start=1):
                segs = segs_by_level[level]
                best_x, best_y = -1.0, None
                best_angle = 0.0
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
                        best_angle = _label_angle(ax, seg, int(idx))
                if best_y is not None:
                    text_str = f"$u_{{{rank}}}$" if self.label_style == "ordinal" else self.ic_label_fmt.format(level)
                    text = ax.annotate(
                        text_str,
                        (best_x, best_y),
                        textcoords="offset points",
                        xytext=(4, 0),
                        color=color_by_level[level],
                        fontsize=9,
                        ha="left",
                        va="center",
                        rotation=best_angle,
                        rotation_mode="anchor",
                        annotation_clip=True,
                    )
                    role = "ic_label" if level in focal_levels else "secondary_ic_label"
                    tag(text, role)
                    tag_attr(text, "_ev_label_default", Label(position=LabelPosition.RIGHT, offset=4))

        if self.show_rays and hasattr(self.func, "utility_type") and self.func.utility_type is UtilityType.KINKED:
            for slope in self.func.ray_slopes():
                draw_ray(ax, slope, x_max, y_max, color=self.ray_color, linewidth=self.ray_linewidth)

        if self.show_kinks and hasattr(self.func, "utility_type") and self.func.utility_type is UtilityType.KINKED:
            for x, y in self.func.kink_points(computed):
                (kink,) = ax.plot(
                    x,
                    y,
                    "o",
                    markersize=self.kink_radius * 4,
                    markerfacecolor=self.kink_color,
                    markeredgecolor=self.kink_color,
                )
                tag(kink, "kink")

        if hasattr(self.func, "subsistence_lines"):
            sub_x, sub_y = self.func.subsistence_lines()
            style = dict(color=self.subsistence_color, linewidth=self.subsistence_linewidth, linestyle="--", alpha=0.6)
            tag(ax.axvline(x=sub_x, **style), "subsistence")
            tag(ax.axhline(y=sub_y, **style), "subsistence")

        return computed

"""Utility-curve renderer for Canvas."""

from __future__ import annotations

from ...canvas.stroke import tag, tag_attr
from ...components.indifference import IndifferenceCurves
from ...enums import LabelPosition
from ...themes.label import Label


def render_utility(
    ax,
    *,
    func,
    levels,
    color: str,
    linewidth: float,
    show_rays: bool,
    ray_color: str,
    ray_linewidth: float,
    show_kinks: bool,
    kink_color: str,
    kink_radius: float,
    label: str | None,
    show_ic_labels: bool,
    ic_label_fmt: str,
    show_bliss: bool,
    bliss_text: str,
    bliss_markersize: float,
    subsistence_color: str,
    subsistence_linewidth: float,
    x_max: float,
    y_max: float,
    **kwargs,
):
    """Render utility contours and optional bliss marker."""
    ic = IndifferenceCurves(
        func,
        levels,
        color=color,
        linewidth=linewidth,
        show_rays=show_rays,
        ray_color=ray_color,
        ray_linewidth=ray_linewidth,
        show_kinks=show_kinks,
        kink_color=kink_color,
        kink_radius=kink_radius,
        label=label,
        show_ic_labels=show_ic_labels,
        ic_label_fmt=ic_label_fmt,
        subsistence_color=subsistence_color,
        subsistence_linewidth=subsistence_linewidth,
    )
    ic.draw(ax, x_max, y_max, **kwargs)

    if show_bliss and hasattr(func, "bliss_x") and hasattr(func, "bliss_y"):
        (bliss,) = ax.plot(
            func.bliss_x,
            func.bliss_y,
            "*",
            color=color,
            markersize=bliss_markersize,
            zorder=5,
        )
        tag(bliss, "bliss")
        text = ax.annotate(
            rf"${bliss_text}$",
            (func.bliss_x, func.bliss_y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=12,
            color=color,
        )
        tag(text, "bliss_label")
        tag_attr(text, "_ev_label_default", Label(position=LabelPosition.TOP_RIGHT, offset=5))
    return ic

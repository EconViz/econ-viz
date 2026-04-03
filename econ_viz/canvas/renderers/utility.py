"""Utility-curve renderer for Canvas."""

from __future__ import annotations

from ...components.indifference import IndifferenceCurves


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
    )
    ic.draw(ax, x_max, y_max, **kwargs)

    if show_bliss and hasattr(func, "bliss_x") and hasattr(func, "bliss_y"):
        ax.plot(
            func.bliss_x,
            func.bliss_y,
            "*",
            color=color,
            markersize=12,
            zorder=5,
        )
        ax.annotate(
            r"$\mathbf{x}^*$",
            (func.bliss_x, func.bliss_y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=12,
            color=color,
        )
    return ic


"""Equilibrium renderer for Canvas."""

from __future__ import annotations

from ...components.equilibrium import EquilibriumPoint


def render_equilibrium(
    ax,
    *,
    eq,
    color: str,
    markersize: float,
    label: str | None,
    drop_dashes: bool,
    show_ray: bool,
    ray_color: str,
    ray_linewidth: float,
    x_max: float,
    y_max: float,
) -> None:
    """Render equilibrium marker/drop-lines/ray."""
    ep = EquilibriumPoint(
        eq,
        color=color,
        markersize=markersize,
        label=label,
        drop_dashes=drop_dashes,
        show_ray=show_ray,
        ray_color=ray_color,
        ray_linewidth=ray_linewidth,
    )
    ep.draw(ax, x_max, y_max)


"""Budget-line renderer for Canvas."""

from __future__ import annotations

from ...components.budget import BudgetConstraint


def render_budget(
    ax,
    *,
    px: float,
    py: float,
    income: float,
    color: str,
    linewidth: float,
    linestyle: str,
    label: str | None,
    fill: bool,
    fill_alpha: float,
    fill_color: str | None = None,
) -> None:
    """Render a budget constraint on the given axes."""
    bc = BudgetConstraint(
        px,
        py,
        income,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        fill=fill,
        fill_alpha=fill_alpha,
        fill_color=fill_color,
    )
    bc.draw(ax)

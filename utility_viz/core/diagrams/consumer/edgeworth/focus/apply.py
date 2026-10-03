"""Equilibrium-focused indifference rendering for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utility_viz.core.diagrams.consumer.edgeworth.focus.selection import curve_count_bounds, should_include_endowment_ic

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox
    from utility_viz.core.diagrams.consumer.edgeworth.config import EquilibriumFocusConfig


def apply_equilibrium_focus(box: EdgeworthBox, px: float, py: float, cfg: EquilibriumFocusConfig) -> None:
    """Draw a bounded number of indifference curves per agent around ``X*``."""
    if box.walrasian_equilibrium is None:
        box.add_walrasian_equilibrium(px=px, py=py)

    min_curves, max_curves = curve_count_bounds(cfg)

    include = cfg.include_endowment_indifference
    if include == "auto":
        include = should_include_endowment_ic(box, min_relative_gap=cfg.min_relative_gap)

    if box.walrasian_equilibrium is None:
        raise ValueError("Walrasian equilibrium is required for equilibrium-focused rendering.")
    x_star, y_star = box.walrasian_equilibrium
    ua_star = box._eval_ua(x_star, y_star)
    ub_star = box._eval_ub(x_star, y_star)

    ua_e, ub_e = _endowment_utilities(box, bool(include))
    target_n = min_curves + (1 if bool(include) and min_curves < max_curves else 0)

    X, Y = box._grid(res=cfg.res)
    U_a = box.utility_a(X, Y)
    U_b = box.utility_b(box.total_x - X, box.total_y - Y)
    ua_levels = _agent_levels(box, ua_star, U_a, target_n, cfg.equilibrium_spread, ua_e)
    ub_levels = _agent_levels(box, ub_star, U_b, target_n, cfg.equilibrium_spread, ub_e)

    lw_eq = cfg.equilibrium_linewidth
    if lw_eq is None:
        lw_eq = cfg.endowment_linewidth
    levels_a = ua_levels[:max_curves]
    levels_b = ub_levels[:max_curves]
    box.equilibrium_focus_levels_a = levels_a
    box.equilibrium_focus_levels_b = levels_b
    box.add_indifference_curves(levels_a=levels_a, levels_b=levels_b, linewidth=lw_eq, res=cfg.res)


def _endowment_utilities(box: EdgeworthBox, include: bool) -> tuple[float | None, float | None]:
    """Both agents' utility at the endowment, when its curves are to be included."""
    if include and box.endowment is not None:
        ex, ey = box.endowment
        return box._eval_ua(ex, ey), box._eval_ub(ex, ey)
    return None, None


def _agent_levels(
    box: EdgeworthBox, anchor: float, utilities: np.ndarray, n: int, spread: float, extra: float | None
) -> list[float]:
    """Indifference levels for one agent around its equilibrium utility."""
    return box._focus_levels(
        anchor=anchor,
        u_min=float(np.nanmin(utilities)),
        u_max=float(np.nanmax(utilities)),
        n=n,
        spread=spread,
        extra=extra,
    )

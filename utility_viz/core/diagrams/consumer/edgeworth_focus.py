"""Equilibrium-focused indifference rendering for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth import EdgeworthBox, EquilibriumFocusConfig


def apply_equilibrium_focus(box: EdgeworthBox, px: float, py: float, cfg: EquilibriumFocusConfig) -> None:
    """Draw a bounded number of indifference curves per agent around ``X*``."""
    if box.walrasian_equilibrium is None:
        box.add_walrasian_equilibrium(px=px, py=py)

    min_curves = max(1, int(cfg.min_curves_per_agent))
    max_curves = max(min_curves, int(cfg.max_curves_per_agent))
    min_curves = max(3, min_curves)
    max_curves = min(5, max_curves)
    if min_curves > max_curves:
        min_curves = max_curves

    include = cfg.include_endowment_indifference
    if include == "auto":
        include = should_include_endowment_ic(box, min_relative_gap=cfg.min_relative_gap)

    if box.walrasian_equilibrium is None:
        raise ValueError("Walrasian equilibrium is required for equilibrium-focused rendering.")
    x_star, y_star = box.walrasian_equilibrium
    ua_star = box._eval_ua(x_star, y_star)
    ub_star = box._eval_ub(x_star, y_star)

    ua_e: float | None = None
    ub_e: float | None = None
    if bool(include) and box.endowment is not None:
        ex, ey = box.endowment
        ua_e = box._eval_ua(ex, ey)
        ub_e = box._eval_ub(ex, ey)

    target_n = min_curves + (1 if bool(include) and min_curves < max_curves else 0)

    X, Y = box._grid(res=cfg.res)
    U_a = box.utility_a(X, Y)
    U_b = box.utility_b(box.total_x - X, box.total_y - Y)
    ua_levels = box._focus_levels(
        anchor=ua_star,
        u_min=float(np.nanmin(U_a)),
        u_max=float(np.nanmax(U_a)),
        n=target_n,
        spread=cfg.equilibrium_spread,
        extra=ua_e if bool(include) else None,
    )
    ub_levels = box._focus_levels(
        anchor=ub_star,
        u_min=float(np.nanmin(U_b)),
        u_max=float(np.nanmax(U_b)),
        n=target_n,
        spread=cfg.equilibrium_spread,
        extra=ub_e if bool(include) else None,
    )

    lw_eq = cfg.equilibrium_linewidth
    if lw_eq is None:
        lw_eq = cfg.endowment_linewidth
    levels_a = ua_levels[:max_curves]
    levels_b = ub_levels[:max_curves]
    box.equilibrium_focus_levels_a = levels_a
    box.equilibrium_focus_levels_b = levels_b
    box.add_indifference_curves(levels_a=levels_a, levels_b=levels_b, linewidth=lw_eq, res=cfg.res)


def should_include_endowment_ic(box: EdgeworthBox, *, min_relative_gap: float) -> bool:
    """Whether the endowment indifference curves differ enough from ``X*``'s."""
    if box.endowment is None or box.walrasian_equilibrium is None:
        return False
    ex, ey = box.endowment
    x_star, y_star = box.walrasian_equilibrium
    ua_e = box._eval_ua(ex, ey)
    ub_e = box._eval_ub(ex, ey)
    ua_s = box._eval_ua(x_star, y_star)
    ub_s = box._eval_ub(x_star, y_star)
    gap_a = abs(ua_s - ua_e) / (abs(ua_s) + 1e-9)
    gap_b = abs(ub_s - ub_e) / (abs(ub_s) + 1e-9)
    return max(gap_a, gap_b) >= max(min_relative_gap, 0.0)

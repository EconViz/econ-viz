"""Which curves the equilibrium-focused rendering keeps."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox
    from utility_viz.core.diagrams.consumer.edgeworth.config import EquilibriumFocusConfig


def curve_count_bounds(cfg: EquilibriumFocusConfig) -> tuple[int, int]:
    """Clamp the per-agent curve count to the supported 3-5 range."""
    min_curves = max(1, int(cfg.min_curves_per_agent))
    max_curves = max(min_curves, int(cfg.max_curves_per_agent))
    min_curves = max(3, min_curves)
    max_curves = min(5, max_curves)
    if min_curves > max_curves:
        min_curves = max_curves
    return min_curves, max_curves


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

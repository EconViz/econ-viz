"""Exchange-theory computations behind the Edgeworth box (equilibrium, core, checks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utility_viz.models.consumer.edgeworth_compute import walrasian_equilibrium_point

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth import EdgeworthBox


def locate_walrasian(box: EdgeworthBox, px: float, py: float) -> tuple[float, float]:
    """Pick the contract-curve point that best satisfies the budget line at ``(px, py)``."""
    if px <= 0 or py <= 0:
        raise ValueError("px and py must be positive.")
    if box.endowment is None:
        raise ValueError("Endowment is required. Call add_endowment(...) first.")
    if len(box.contract_curve_points) == 0:
        box.add_contract_curve()

    ex, ey = box.endowment
    income = px * ex + py * ey
    return walrasian_equilibrium_point(
        candidates=box.contract_curve_points,
        px=px,
        py=py,
        income=income,
        mrs_a_fn=lambda x, y: box._mrs(box.utility_a, x, y),
        mrs_b_fn=lambda x, y: box._mrs(box.utility_b, box.total_x - x, box.total_y - y),
    )


def core_allocations(box: EdgeworthBox, tol: float) -> list[tuple[float, float]]:
    """Contract-curve points that leave both agents at least as well off as at the endowment."""
    if box.endowment is None:
        raise ValueError("Endowment is required. Call add_endowment(...) first.")
    if len(box.contract_curve_points) == 0:
        raise ValueError("Contract curve is required. Call add_contract_curve(...) first.")

    ex, ey = box.endowment
    ua_e = box._eval_ua(ex, ey)
    ub_e = box._eval_ub(ex, ey)

    core: list[tuple[float, float]] = []
    for x, y in box.contract_curve_points:
        if box._eval_ua(float(x), float(y)) >= ua_e - tol and box._eval_ub(float(x), float(y)) >= ub_e - tol:
            core.append((float(x), float(y)))
    return core


def check_allocation(
    box: EdgeworthBox, x: float, y: float, px: float | None, py: float | None, tol: float
) -> dict[str, bool]:
    """Return key checklist conditions at a candidate allocation."""
    checks: dict[str, bool] = {}
    checks["market_clearing"] = abs((x + (box.total_x - x)) - box.total_x) <= tol

    if box.endowment is not None:
        ex, ey = box.endowment
        checks["individual_rationality"] = (
            box._eval_ua(x, y) >= box._eval_ua(ex, ey) - tol and box._eval_ub(x, y) >= box._eval_ub(ex, ey) - tol
        )
    else:
        checks["individual_rationality"] = False

    if px is not None and py is not None and box.endowment is not None:
        ex, ey = box.endowment
        income = px * ex + py * ey
        checks["budget_balance"] = abs(px * x + py * y - income) <= tol * max(income, 1.0)
    else:
        checks["budget_balance"] = False

    mrs_a = box._mrs(box.utility_a, x, y)
    mrs_b = box._mrs(box.utility_b, box.total_x - x, box.total_y - y)
    checks["mrs_equal"] = (
        np.isfinite(mrs_a) and np.isfinite(mrs_b) and mrs_a > 0 and mrs_b > 0 and abs(np.log(mrs_a / mrs_b)) <= 0.08
    )
    return checks

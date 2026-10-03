"""Checklist of first-welfare-theorem conditions at a candidate allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


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

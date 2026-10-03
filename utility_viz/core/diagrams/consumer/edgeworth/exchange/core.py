"""Core allocations for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


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

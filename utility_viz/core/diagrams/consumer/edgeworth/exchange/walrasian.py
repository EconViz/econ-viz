"""Walrasian equilibrium location for the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utility_viz.models.consumer.edgeworth_compute import walrasian_equilibrium_point

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


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

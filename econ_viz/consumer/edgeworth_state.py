"""State container for Edgeworth box workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EdgeworthState:
    """Mutable state used while composing an Edgeworth diagram."""

    endowment: tuple[float, float] | None = None
    contract_curve_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    core_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    walrasian_equilibrium: tuple[float, float] | None = None
    equilibrium_focus_levels_a: list[float] = field(default_factory=list)
    equilibrium_focus_levels_b: list[float] = field(default_factory=list)


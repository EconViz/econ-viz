"""Configuration for equilibrium-focused Edgeworth rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquilibriumFocusConfig:
    """Configuration for equilibrium-focused indifference rendering."""

    include_endowment_indifference: bool | str = "auto"
    min_relative_gap: float = 0.2
    min_curves_per_agent: int = 3
    max_curves_per_agent: int = 5
    equilibrium_spread: float = 0.35
    equilibrium_linewidth: float | None = None
    endowment_linewidth: float | None = None
    res: int = 300

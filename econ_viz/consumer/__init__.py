"""Consumer-choice sweeps and derived teaching diagrams."""

from .paths import ConsumptionPath, IncomePath, LinearBudget, PricePath
from .demand import DemandDiagram
from .edgeworth import EdgeworthBox, EquilibriumFocusConfig

__all__ = [
    "ConsumptionPath",
    "IncomePath",
    "LinearBudget",
    "PricePath",
    "DemandDiagram",
    "EdgeworthBox",
    "EquilibriumFocusConfig",
]

"""Exchange-theory computations behind the Edgeworth box (equilibrium, core, checks)."""

from utility_viz.core.diagrams.consumer.edgeworth.exchange.checks import check_allocation
from utility_viz.core.diagrams.consumer.edgeworth.exchange.core import core_allocations
from utility_viz.core.diagrams.consumer.edgeworth.exchange.walrasian import locate_walrasian

__all__ = ["check_allocation", "core_allocations", "locate_walrasian"]

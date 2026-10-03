"""
econ_viz.models.optimization — Equilibrium solvers for consumer choice problems.

Given a :class:`~econ_viz.models.utility.protocol.UtilityFunction` and a budget
constraint, the solver finds the optimal consumption bundle (tangency
interior solution or corner solution) and returns a structured result
that the :class:`~econ_viz.core.canvas.base.Canvas` can render directly.
"""

from econ_viz.models.optimization.analytic import solution_tex
from econ_viz.models.optimization.comparative import ComparativeStatics, comparative_statics
from econ_viz.models.optimization.decomposition import (
    DecompositionMethod,
    PriceEffectDecomposition,
    decompose_price_effect,
)
from econ_viz.models.optimization.slutsky import SlutskyMatrix, slutsky_matrix
from econ_viz.models.optimization.solver import Equilibrium, solve

__all__ = [
    "Equilibrium",
    "solve",
    "solution_tex",
    "ComparativeStatics",
    "DecompositionMethod",
    "PriceEffectDecomposition",
    "SlutskyMatrix",
    "comparative_statics",
    "decompose_price_effect",
    "slutsky_matrix",
]

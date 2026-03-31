"""
econ_viz.optimizer — Equilibrium solvers for consumer choice problems.

Given a :class:`~econ_viz.models.protocol.UtilityFunction` and a budget
constraint, the solver finds the optimal consumption bundle (tangency
interior solution or corner solution) and returns a structured result
that the :class:`~econ_viz.canvas.base.Canvas` can render directly.
"""

from .analytic import solution_tex
from .comparative import ComparativeStatics, comparative_statics
from .solver import Equilibrium, solve
from .slutsky import SlutskyMatrix, slutsky_matrix

__all__ = [
    "Equilibrium",
    "solve",
    "solution_tex",
    "ComparativeStatics",
    "SlutskyMatrix",
    "comparative_statics",
    "slutsky_matrix",
]

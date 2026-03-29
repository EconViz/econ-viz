"""
econ_viz.optimizer — Equilibrium solvers for consumer choice problems.

Given a :class:`~econ_viz.models.protocol.UtilityFunction` and a budget
constraint, the solver finds the optimal consumption bundle (tangency
interior solution or corner solution) and returns a structured result
that the :class:`~econ_viz.canvas.base.Canvas` can render directly.
"""

from .solver import Equilibrium, solve
from .statics import ComparativeStatics, comparative_statics

__all__ = ["Equilibrium", "solve", "ComparativeStatics", "comparative_statics"]

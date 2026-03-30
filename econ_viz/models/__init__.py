"""
econ_viz.models — Parametric utility function specifications.

Each class in this subpackage represents a family of utility functions
commonly encountered in consumer theory. Every model conforms to the
:class:`~econ_viz.models.protocol.UtilityFunction` protocol: it is a
callable dataclass that evaluates U(x, y) element-wise over NumPy arrays
and exposes ``utility_type``, ``ray_slopes``, and ``kink_points`` for
rendering support.
"""

from .core import CobbDouglas, Leontief, PerfectSubstitutes, CES, Satiation, QuasiLinear, StoneGeary, Translog
from .advanced import CustomUtility, MultiGoodCD
from .protocol import UtilityFunction

__all__ = [
    "CobbDouglas",
    "Leontief",
    "PerfectSubstitutes",
    "CES",
    "Satiation",
    "QuasiLinear",
    "CustomUtility",
    "MultiGoodCD",
    "StoneGeary",
    "Translog",
    "UtilityFunction",
]

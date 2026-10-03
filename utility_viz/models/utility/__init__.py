"""
utility_viz.models.utility — Parametric utility function specifications.

Each class in this subpackage represents a family of utility functions
commonly encountered in consumer theory. Every model conforms to the
:class:`~utility_viz.models.utility.protocol.UtilityFunction` protocol: it is a
callable dataclass that evaluates U(x, y) element-wise over NumPy arrays
and exposes ``utility_type``, ``ray_slopes``, and ``kink_points`` for
rendering support.
"""

from utility_viz.models.utility.advanced import CustomUtility, MultiGoodCD
from utility_viz.models.utility.core import (
    CES,
    CobbDouglas,
    Haagsma,
    Leontief,
    PerfectSubstitutes,
    QuasiLinear,
    Satiation,
    StoneGeary,
    Translog,
)
from utility_viz.models.utility.parser import parse_latex
from utility_viz.models.utility.protocol import UtilityFunction
from utility_viz.models.utility.registry import build_registered_model, get_model_registry

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
    "Haagsma",
    "UtilityFunction",
    "parse_latex",
    "get_model_registry",
    "build_registered_model",
]

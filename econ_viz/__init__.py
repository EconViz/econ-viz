"""
econ_viz — A toolkit for producing publication-quality economic diagrams.

This package provides a declarative interface for constructing standard
microeconomic visualizations (indifference curves, budget constraints, etc.)
on a configurable canvas. Figures can be exported as raster images or as
vector graphics for publication and web workflows.
"""

import sys

from .canvas.base import Canvas
from .canvas.figure import Figure
from .canvas.layers import Layer
from .analysis import levels
from . import analysis
from .enums import UtilityType, ExportFormat, Layout
from .exceptions import EconVizError, ExportError, InvalidParameterError, OptimizationError, ParseError
from . import themes
from .themes.theme import Theme
from .optimizer import (
    ComparativeStatics,
    Equilibrium,
    SlutskyMatrix,
    comparative_statics,
    solve,
    solution_tex,
    slutsky_matrix,
)
from .components import IndifferenceCurves, BudgetConstraint, EquilibriumPoint
from .models import parse_latex
from .models.advanced import CustomUtility, MultiGoodCD
from .consumer import (
    ConsumptionPath,
    DemandDiagram,
    EdgeworthBox,
    EquilibriumFocusConfig,
    IncomePath,
    LinearBudget,
    PricePath,
)
from .utils.logging import get_logger

sys.modules.setdefault(__name__ + ".figure", sys.modules[__name__ + ".canvas.figure"])
sys.modules.setdefault(__name__ + ".levels", sys.modules[__name__ + ".analysis.levels"])
sys.modules.setdefault(__name__ + ".parser", sys.modules[__name__ + ".models.parser"])
sys.modules.setdefault(__name__ + ".logging", sys.modules[__name__ + ".utils.logging"])

__all__ = [
    "Canvas",
    "Layer",
    "UtilityType",
    "ExportFormat",
    "Layout",
    "EconVizError",
    "OptimizationError",
    "InvalidParameterError",
    "ExportError",
    "ParseError",
    "levels",
    "analysis",
    "themes",
    "Theme",
    "Equilibrium",
    "solve",
    "solution_tex",
    "ComparativeStatics",
    "comparative_statics",
    "SlutskyMatrix",
    "slutsky_matrix",
    "IndifferenceCurves",
    "BudgetConstraint",
    "EquilibriumPoint",
    "parse_latex",
    "CustomUtility",
    "MultiGoodCD",
    "Figure",
    "ConsumptionPath",
    "LinearBudget",
    "PricePath",
    "IncomePath",
    "DemandDiagram",
    "EdgeworthBox",
    "EquilibriumFocusConfig",
    "get_logger",
]

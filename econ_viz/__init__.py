"""
econ_viz — A toolkit for producing publication-quality economic diagrams.

This package provides a declarative interface for constructing standard
microeconomic visualizations (indifference curves, budget constraints, etc.)
on a configurable canvas. Figures can be exported as raster images or as
TikZ/PGFPlots source for direct inclusion in LaTeX documents.
"""

from .canvas.base import Canvas
from .canvas.layers import Layer
from .enums import UtilityType, ExportFormat
from .exceptions import EconVizError, OptimizationError, InvalidParameterError, ExportError
from . import levels
from . import themes
from .themes.theme import Theme
from .optimizer import Equilibrium, solve
from .components import IndifferenceCurves, BudgetConstraint, EquilibriumPoint
from .parser import parse_latex
from .models.advanced import CustomUtility, MultiGoodCD

__all__ = [
    "Canvas",
    "Layer",
    "UtilityType",
    "ExportFormat",
    "EconVizError",
    "OptimizationError",
    "InvalidParameterError",
    "ExportError",
    "levels",
    "themes",
    "Theme",
    "Equilibrium",
    "solve",
    "IndifferenceCurves",
    "BudgetConstraint",
    "EquilibriumPoint",
    "parse_latex",
    "CustomUtility",
    "MultiGoodCD",
]

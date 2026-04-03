"""
econ_viz — A toolkit for producing publication-quality economic diagrams.

This package provides a declarative interface for constructing standard
microeconomic visualizations (indifference curves, budget constraints, etc.)
on a configurable canvas. Figures can be exported as raster images or as
vector graphics for publication and web workflows.
"""

from __future__ import annotations

import importlib
import sys

_MODULE_EXPORTS = {
    "levels": "econ_viz.analysis.levels",
    "analysis": "econ_viz.analysis",
    "themes": "econ_viz.themes",
}

_ATTR_EXPORTS = {
    "Canvas": ("econ_viz.canvas.base", "Canvas"),
    "Figure": ("econ_viz.canvas.figure", "Figure"),
    "Layer": ("econ_viz.canvas.layers", "Layer"),
    "UtilityType": ("econ_viz.enums", "UtilityType"),
    "ExportFormat": ("econ_viz.enums", "ExportFormat"),
    "Layout": ("econ_viz.enums", "Layout"),
    "EconVizError": ("econ_viz.exceptions", "EconVizError"),
    "ExportError": ("econ_viz.exceptions", "ExportError"),
    "InvalidParameterError": ("econ_viz.exceptions", "InvalidParameterError"),
    "OptimizationError": ("econ_viz.exceptions", "OptimizationError"),
    "ParseError": ("econ_viz.exceptions", "ParseError"),
    "Theme": ("econ_viz.themes.theme", "Theme"),
    "Equilibrium": ("econ_viz.optimizer", "Equilibrium"),
    "solve": ("econ_viz.optimizer", "solve"),
    "solution_tex": ("econ_viz.optimizer", "solution_tex"),
    "ComparativeStatics": ("econ_viz.optimizer", "ComparativeStatics"),
    "comparative_statics": ("econ_viz.optimizer", "comparative_statics"),
    "SlutskyMatrix": ("econ_viz.optimizer", "SlutskyMatrix"),
    "slutsky_matrix": ("econ_viz.optimizer", "slutsky_matrix"),
    "IndifferenceCurves": ("econ_viz.components", "IndifferenceCurves"),
    "BudgetConstraint": ("econ_viz.components", "BudgetConstraint"),
    "EquilibriumPoint": ("econ_viz.components", "EquilibriumPoint"),
    "parse_latex": ("econ_viz.models", "parse_latex"),
    "CustomUtility": ("econ_viz.models.advanced", "CustomUtility"),
    "MultiGoodCD": ("econ_viz.models.advanced", "MultiGoodCD"),
    "ConsumptionPath": ("econ_viz.consumer", "ConsumptionPath"),
    "LinearBudget": ("econ_viz.consumer", "LinearBudget"),
    "PricePath": ("econ_viz.consumer", "PricePath"),
    "IncomePath": ("econ_viz.consumer", "IncomePath"),
    "DemandDiagram": ("econ_viz.consumer", "DemandDiagram"),
    "EdgeworthBox": ("econ_viz.consumer", "EdgeworthBox"),
    "EquilibriumFocusConfig": ("econ_viz.consumer", "EquilibriumFocusConfig"),
    "EdgeworthState": ("econ_viz.consumer", "EdgeworthState"),
    "get_logger": ("econ_viz.utils.logging", "get_logger"),
}

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
    "EdgeworthState",
    "get_logger",
]


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        mod = importlib.import_module(_MODULE_EXPORTS[name])
        globals()[name] = mod
        return mod
    if name in _ATTR_EXPORTS:
        module_name, attr_name = _ATTR_EXPORTS[name]
        mod = importlib.import_module(module_name)
        value = getattr(mod, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


sys.modules.setdefault(__name__ + ".figure", importlib.import_module("econ_viz.canvas.figure"))
sys.modules.setdefault(__name__ + ".levels", importlib.import_module("econ_viz.analysis.levels"))
sys.modules.setdefault(__name__ + ".parser", importlib.import_module("econ_viz.models.parser"))
sys.modules.setdefault(__name__ + ".logging", importlib.import_module("econ_viz.utils.logging"))


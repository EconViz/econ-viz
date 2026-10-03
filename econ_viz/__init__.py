"""
econ_viz — A toolkit for producing publication-quality economic diagrams.

This package provides a declarative interface for constructing standard
microeconomic visualizations (indifference curves, budget constraints, etc.)
on a configurable canvas. Figures can be exported as raster images or as
vector graphics for publication and web workflows.
"""

from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "levels": "econ_viz.models.analysis.levels",
    "analysis": "econ_viz.models.analysis",
    "themes": "econ_viz.core.themes",
}

_ATTR_EXPORTS = {
    "Canvas": ("econ_viz.core.canvas.base", "Canvas"),
    "Figure": ("econ_viz.core.layout.figure", "Figure"),
    "Layer": ("econ_viz.models.curves.layers", "Layer"),
    "UtilityType": ("econ_viz.enums", "UtilityType"),
    "ExportFormat": ("econ_viz.enums", "ExportFormat"),
    "Layout": ("econ_viz.enums", "Layout"),
    "ArrowStyle": ("econ_viz.enums", "ArrowStyle"),
    "LabelPosition": ("econ_viz.enums", "LabelPosition"),
    "LineStyle": ("econ_viz.enums", "LineStyle"),
    "Stroke": ("econ_viz.core.styles.stroke", "Stroke"),
    "Axis": ("econ_viz.core.styles.axis", "Axis"),
    "Config": ("econ_viz.core.config.settings", "Config"),
    "Fill": ("econ_viz.core.styles.fill", "Fill"),
    "Label": ("econ_viz.core.styles.label", "Label"),
    "Legend": ("econ_viz.core.styles.legend", "Legend"),
    "Marker": ("econ_viz.core.styles.marker", "Marker"),
    "Effect": ("econ_viz.core.rendering.effect", "Effect"),
    "EconVizError": ("econ_viz.core.errors.exceptions", "EconVizError"),
    "ExportError": ("econ_viz.core.errors.exceptions", "ExportError"),
    "InvalidParameterError": ("econ_viz.core.errors.exceptions", "InvalidParameterError"),
    "OptimizationError": ("econ_viz.core.errors.exceptions", "OptimizationError"),
    "ParseError": ("econ_viz.core.errors.exceptions", "ParseError"),
    "Theme": ("econ_viz.core.themes.theme", "Theme"),
    "Equilibrium": ("econ_viz.models.optimization", "Equilibrium"),
    "solve": ("econ_viz.models.optimization", "solve"),
    "solution_tex": ("econ_viz.models.optimization", "solution_tex"),
    "ComparativeStatics": ("econ_viz.models.optimization", "ComparativeStatics"),
    "DecompositionMethod": ("econ_viz.models.optimization", "DecompositionMethod"),
    "PriceEffectDecomposition": ("econ_viz.models.optimization", "PriceEffectDecomposition"),
    "comparative_statics": ("econ_viz.models.optimization", "comparative_statics"),
    "decompose_price_effect": ("econ_viz.models.optimization", "decompose_price_effect"),
    "SlutskyMatrix": ("econ_viz.models.optimization", "SlutskyMatrix"),
    "slutsky_matrix": ("econ_viz.models.optimization", "slutsky_matrix"),
    "IndifferenceCurves": ("econ_viz.core.diagrams.components", "IndifferenceCurves"),
    "BudgetConstraint": ("econ_viz.core.diagrams.components", "BudgetConstraint"),
    "EquilibriumPoint": ("econ_viz.core.diagrams.components", "EquilibriumPoint"),
    "parse_latex": ("econ_viz.models.utility", "parse_latex"),
    "CustomUtility": ("econ_viz.models.utility.advanced", "CustomUtility"),
    "MultiGoodCD": ("econ_viz.models.utility.advanced", "MultiGoodCD"),
    "ConsumptionPath": ("econ_viz.models.consumer", "ConsumptionPath"),
    "LinearBudget": ("econ_viz.models.consumer", "LinearBudget"),
    "PricePath": ("econ_viz.models.consumer", "PricePath"),
    "IncomePath": ("econ_viz.models.consumer", "IncomePath"),
    "DemandDiagram": ("econ_viz.core.diagrams.consumer", "DemandDiagram"),
    "EdgeworthBox": ("econ_viz.core.diagrams.consumer", "EdgeworthBox"),
    "EquilibriumFocusConfig": ("econ_viz.core.diagrams.consumer", "EquilibriumFocusConfig"),
    "EdgeworthState": ("econ_viz.models.consumer", "EdgeworthState"),
    "get_logger": ("econ_viz.utils.logging", "get_logger"),
}

__all__ = [
    "Canvas",
    "Layer",
    "UtilityType",
    "ExportFormat",
    "Layout",
    "ArrowStyle",
    "LabelPosition",
    "LineStyle",
    "Stroke",
    "Axis",
    "Config",
    "Fill",
    "Label",
    "Legend",
    "Marker",
    "Effect",
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
    "DecompositionMethod",
    "PriceEffectDecomposition",
    "comparative_statics",
    "decompose_price_effect",
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

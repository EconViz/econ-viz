"""Root facade and models facade import contract (#131)."""

from __future__ import annotations

import importlib

import pytest

import econ_viz
import econ_viz.models as models_facade

AGREED_ROOT_EXPORTS = [
    # canvas / layout / configuration
    "Canvas", "Figure", "Config", "Layer",
    # themes and style objects
    "themes", "Theme", "Stroke", "Fill", "Label", "Legend", "Marker", "Axis", "Effect",
    # enums
    "UtilityType", "ExportFormat", "Layout", "ArrowStyle", "LabelPosition", "LineStyle",
    # public exceptions
    "EconVizError", "ExportError", "InvalidParameterError", "OptimizationError", "ParseError",
    # diagrams
    "IndifferenceCurves", "BudgetConstraint", "EquilibriumPoint", "DemandDiagram", "EdgeworthBox",
    # economics (also available from econ_viz.models)
    "Equilibrium", "solve", "parse_latex", "CustomUtility", "MultiGoodCD",
]  # fmt: skip


@pytest.mark.parametrize("name", AGREED_ROOT_EXPORTS)
def test_agreed_root_export_resolves(name):
    assert hasattr(econ_viz, name)
    assert name in econ_viz.__all__


def test_every_name_in_all_resolves():
    for name in econ_viz.__all__:
        assert getattr(econ_viz, name) is not None, name


def test_root_dir_lists_public_names():
    assert set(econ_viz.__all__) <= set(dir(econ_viz))


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        econ_viz.DefinitelyNotAThing  # noqa: B018


def test_root_import_is_lazy():
    """Importing the root must not eagerly import heavy subpackages."""
    import subprocess
    import sys

    code = "import sys, econ_viz; sys.exit(0 if 'econ_viz.core.canvas.base' not in sys.modules else 1)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.parametrize("name", models_facade.__all__)
def test_models_facade_exports_resolve(name):
    assert getattr(models_facade, name) is not None


def test_models_facade_has_no_drawing_dependency():
    import subprocess
    import sys

    code = "import sys, econ_viz.models; sys.exit(1 if any(m.startswith('econ_viz.core.canvas') for m in sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.parametrize(
    "module",
    [
        "econ_viz.core.canvas", "econ_viz.core.layout", "econ_viz.core.styles.stroke", "econ_viz.core.themes",
        "econ_viz.core.config", "econ_viz.core.errors", "econ_viz.core.export", "econ_viz.core.constants",
        "econ_viz.core.rendering.stroke", "econ_viz.core.diagrams.components", "econ_viz.core.diagrams.consumer",
        "econ_viz.core.animation", "econ_viz.core.interactive", "econ_viz.models.utility", "econ_viz.models.curves",
        "econ_viz.models.consumer", "econ_viz.models.optimization", "econ_viz.models.analysis",
        "econ_viz.enums", "econ_viz.utils", "econ_viz.cli",
    ],
)  # fmt: skip
def test_subpackages_import(module):
    importlib.import_module(module)

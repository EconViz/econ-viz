"""The ``econ_viz`` 1.x compatibility package (#132): documented imports and behaviour."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import utility_viz
from utility_viz import UtilityVizDeprecationWarning

# --- 1.x documented imports (taken from the 1.12 README and examples) -------------------------------

ROOT_NAMES_SAME_OBJECT = [
    "Stroke", "Fill", "Label", "Legend", "Marker", "Axis", "Effect", "Theme", "Config", "Layer",
    "UtilityType", "ExportFormat", "ArrowStyle", "LabelPosition", "LineStyle",
    "EconVizError", "ExportError", "InvalidParameterError", "OptimizationError", "ParseError",
    "Equilibrium", "solve", "solution_tex", "slutsky_matrix", "SlutskyMatrix", "comparative_statics",
    "ComparativeStatics", "decompose_price_effect", "DecompositionMethod", "PriceEffectDecomposition",
    "IndifferenceCurves", "BudgetConstraint", "EquilibriumPoint", "parse_latex", "CustomUtility", "MultiGoodCD",
    "ConsumptionPath", "LinearBudget", "PricePath", "IncomePath", "DemandDiagram", "EdgeworthBox",
    "EquilibriumFocusConfig", "EdgeworthState", "get_logger", "levels", "analysis", "themes",
]  # fmt: skip


@pytest.mark.parametrize("name", ROOT_NAMES_SAME_OBJECT)
def test_unchanged_root_names_are_the_same_objects(name):
    import econ_viz

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # these must not warn
        assert getattr(econ_viz, name) is getattr(utility_viz, name)


def test_all_of_1x_root_all_is_importable():
    import econ_viz

    for name in econ_viz.__all__:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UtilityVizDeprecationWarning)
            assert getattr(econ_viz, name) is not None, name


def test_unknown_attribute_raises():
    import econ_viz

    with pytest.raises(AttributeError):
        econ_viz.DefinitelyMissing  # noqa: B018


@pytest.mark.parametrize(
    ("legacy", "attr"),
    [
        ("econ_viz.models", "CobbDouglas"),
        ("econ_viz.models", "CES"),
        ("econ_viz.models", "parse_latex"),
        ("econ_viz.models.advanced", "CustomUtility"),
        ("econ_viz.models.core", "Leontief"),
        ("econ_viz.models.parser", "parse_latex"),
        ("econ_viz.optimizer", "decompose_price_effect"),
        ("econ_viz.optimizer", "DecompositionMethod"),
        ("econ_viz.optimizer.solver", "solve"),
        ("econ_viz.analysis", "HomogeneityAnalyzer"),
        ("econ_viz.themes", "dark"),
        ("econ_viz.themes.theme", "Theme"),
        ("econ_viz.themes.stroke", "Stroke"),
        ("econ_viz.consumer", "EdgeworthBox"),
        ("econ_viz.consumer", "PricePath"),
        ("econ_viz.consumer.paths", "LinearBudget"),
        ("econ_viz.io.backend_tikz", "figure_to_tikz"),
        ("econ_viz.canvas.layers", "Layer"),
        ("econ_viz.canvas.base", "Canvas"),
        ("econ_viz.canvas", "Layer"),
        ("econ_viz.enums", "UtilityType"),
        ("econ_viz.exceptions", "EconVizError"),
        ("econ_viz.config", "Config"),
        ("econ_viz.contours", "around_anchor_levels"),
        ("econ_viz.components", "BudgetConstraint"),
        ("econ_viz.levels", "around"),
        ("econ_viz.parser", "parse_latex"),
        ("econ_viz.logging", "get_logger"),
    ],
)
def test_documented_legacy_module_paths_resolve(legacy, attr):
    module = importlib.import_module(legacy)
    assert hasattr(module, attr)


def test_from_import_of_legacy_modules_and_models():
    from econ_viz.models import CES, CobbDouglas

    from econ_viz import levels, themes  # noqa: F401
    from utility_viz.models import CES as NEW_CES
    from utility_viz.models import CobbDouglas as NEW_CD

    assert CobbDouglas is NEW_CD and CES is NEW_CES


def test_legacy_module_alias_keeps_real_spec():
    """Aliasing must not clobber the real module's ``__spec__``/``__name__``."""
    import econ_viz.models

    assert econ_viz.models.__name__ == "utility_viz.models"
    assert econ_viz.models.__spec__.name == "utility_viz.models"


def test_widget_viewer_path():
    pytest.importorskip("ipywidgets")
    from econ_viz.interactive import WidgetViewer

    assert WidgetViewer.__name__ == "WidgetViewer"


# --- deprecation behaviour ---------------------------------------------------------------------------


def _record(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [w for w in caught if issubclass(w.category, UtilityVizDeprecationWarning)]


def test_legacy_canvas_warns_once_per_construction_with_details():
    import econ_viz

    canvas, caught = _record(lambda: econ_viz.Canvas(x_max=5, y_max=5))
    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, FutureWarning)
    text = str(w.message)
    assert "deprecated since 2.0.0" in text
    assert "removed in 3.0.0" in text
    assert "utility_viz.Canvas" in text
    assert w.filename == __file__  # stacklevel points at the caller
    # further use of the object does not warn again
    _, more = _record(lambda: canvas.add_budget(1, 1, 5))
    assert more == []
    assert isinstance(canvas, utility_viz.Canvas)


def test_legacy_canvas_still_renders(tmp_path):
    from econ_viz.models import CobbDouglas

    import econ_viz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UtilityVizDeprecationWarning)
        cvs = econ_viz.Canvas(x_max=10, y_max=10)
        cvs.add_utility(CobbDouglas(alpha=0.5, beta=0.5), levels=[2, 4])
        eq = econ_viz.solve(CobbDouglas(alpha=0.5, beta=0.5), px=1, py=1, income=8)
        cvs.add_equilibrium(eq)
        out = tmp_path / "legacy.png"
        cvs.save(str(out))
    assert out.stat().st_size > 0


def test_legacy_figure_warns_with_planned_replacement():
    import econ_viz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UtilityVizDeprecationWarning)
        layout = econ_viz.Layout.SIDE_BY_SIDE
    fig, caught = _record(lambda: econ_viz.Figure(layout, x_max=5, y_max=5))
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "deprecated since 2.0.0" in text and "removed in 3.0.0" in text
    assert "utility_viz.Figure" in text and "CanvasGrid" in text and "planned" in text
    assert isinstance(fig, utility_viz.Figure)
    assert caught[0].filename == __file__


def test_legacy_layout_access_warns_and_maps_to_current_enum():
    import econ_viz

    layout, caught = _record(lambda: econ_viz.Layout)
    assert layout is utility_viz.Layout
    assert len(caught) == 1
    assert "utility_viz.Layout" in str(caught[0].message)
    assert "removed in 3.0.0" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_legacy_animator_warns_and_works(tmp_path):
    pytest.importorskip("PIL")
    from econ_viz.animation import Animator

    def factory(v):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UtilityVizDeprecationWarning)
            return utility_viz.Canvas(x_max=10, y_max=10).add_budget(1, 1, v)

    anim, caught = _record(lambda: Animator(factory, frames=np.linspace(2, 6, 3)))
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "Animator" in text and "planned" in text and "Animation" in text
    assert "deprecated since 2.0.0" in text and "removed in 3.0.0" in text
    anim.save(tmp_path / "a.gif", fps=2, dpi=40)
    assert (tmp_path / "a.gif").exists()


def test_legacy_shims_are_cached_and_consistent():
    import econ_viz
    import econ_viz.canvas
    import econ_viz.figure

    assert econ_viz.Canvas is econ_viz.canvas.Canvas
    assert econ_viz.Figure is econ_viz.canvas.Figure is econ_viz.figure.Figure
    assert issubclass(econ_viz.Canvas, utility_viz.Canvas)


def test_utility_viz_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utility_viz.Canvas(x_max=5, y_max=5)
        utility_viz.Figure(utility_viz.Layout.SIDE_BY_SIDE)


def test_importing_legacy_package_does_not_warn_by_itself():
    code = (
        "import warnings; warnings.simplefilter('error'); import econ_viz, econ_viz.models; "
        "from econ_viz import Stroke, solve"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

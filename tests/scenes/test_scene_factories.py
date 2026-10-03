import numpy as np
import pytest
from mosaickit import FillLayer, MarkerLayer, PathLayer, Stroke, TextLayer

from utility_viz.core.errors.exceptions import InvalidParameterError
from utility_viz.core.scenes import (
    Budget,
    Equilibrium,
    Indifference,
    budget_layers,
    equilibrium_layers,
    indifference_layers,
)
from utility_viz.models import CES, CobbDouglas, Leontief, PerfectSubstitutes, QuasiLinear, StoneGeary, solve
from utility_viz.models.curves import sample_path, trace_level_sets


def test_budget_layers_geometry_and_ids():
    (line,) = budget_layers(2, 3, 30)
    assert isinstance(line, PathLayer)
    assert line.id == "budget" and line.role == "utility.budget"
    assert line.path == ((15.0, 0.0), (0.0, 10.0))


def test_budget_fill_and_label_layers():
    fill, line, label = budget_layers(2, 3, 30, fill=True, label="I = p_x x + p_y y")
    assert isinstance(fill, FillLayer) and fill.id == "budget.fill" and fill.role == Budget.FILL
    assert fill.z_index < line.z_index
    assert isinstance(label, TextLayer) and label.math and label.id == "budget.label"


def test_budget_compensated_role_and_override():
    (line,) = budget_layers(1, 1, 5, layer_id="b2", role=Budget.COMPENSATED, stroke=Stroke(width=3))
    assert line.role == "utility.budget.compensated" and line.stroke.width == 3


@pytest.mark.parametrize("args", [(0, 1, 1), (1, -1, 1), (1, 1, 0)])
def test_budget_rejects_nonpositive(args):
    with pytest.raises(InvalidParameterError):
        budget_layers(*args)


def test_equilibrium_layers():
    eq = solve(CobbDouglas(0.5, 0.5), 2.0, 3.0, 30.0)
    marker, label, drop, ray = equilibrium_layers(eq, ray_to=(15, 10))
    assert isinstance(marker, MarkerLayer) and marker.points == ((eq.x, eq.y),)
    assert marker.model is eq and marker.role == Equilibrium.MAIN
    assert isinstance(label, TextLayer) and label.id == "equilibrium.label"
    assert drop.path == ((0.0, eq.y), (eq.x, eq.y), (eq.x, 0.0))
    assert drop.role == Equilibrium.DROP
    (x1, y1) = ray.path[1]
    assert y1 / x1 == pytest.approx(eq.y / eq.x)
    assert x1 <= 15 + 1e-9 and y1 <= 10 + 1e-9 and (x1 == pytest.approx(15) or y1 == pytest.approx(10))


def test_equilibrium_optional_layers_omitted():
    eq = solve(CobbDouglas(0.5, 0.5), 2.0, 3.0, 30.0)
    layers = equilibrium_layers(eq, label=None, drop_lines=False)
    assert [layer.id for layer in layers] == ["equilibrium"]


@pytest.mark.parametrize(
    ("func", "level", "tol"),
    [
        (CobbDouglas(0.4, 0.6), 4.0, 0.02),
        (CES(), 5.0, 0.02),
        (QuasiLinear(), 6.0, 0.1),
        (PerfectSubstitutes(), 12.0, 1e-9),
        (Leontief(), 3.5, 0.05),
        (StoneGeary(), 5.0, 0.05),
    ],
)
def test_trace_level_sets_follows_the_level_set(func, level, tol):
    (path,) = trace_level_sets(func, [level], (0.1, 15.0), (0.1, 10.0)).for_level(level)
    xs, ys = np.array(sample_path(path)).T
    assert np.abs(np.asarray(func(xs, ys)) - level).max() < tol


def test_trace_level_sets_empty_for_absent_level():
    assert trace_level_sets(CobbDouglas(), [1e6], (0.1, 12.0), (0.1, 12.0)).for_level(1e6) == ()


def test_indifference_layer_keeps_source_bezier():
    from bezierkit.bezier.path import PiecewiseBezier

    layers, _ = indifference_layers(CobbDouglas(), [4.0], 12, 12)
    (layer,) = layers
    assert isinstance(layer.model, PiecewiseBezier)


def test_indifference_layers_levels_ids_and_labels():
    func = CobbDouglas(0.5, 0.5)
    layers, levels = indifference_layers(func, 3, 15, 10, show_labels=True)
    assert len(levels) == 3 and levels == sorted(levels)
    ids = [layer.id for layer in layers]
    assert ids == ["ic.1", "ic.1.label", "ic.2", "ic.2.label", "ic.3", "ic.3.label"], ids
    assert all(layer.role == Indifference.MAIN for layer in layers if isinstance(layer, PathLayer))
    first_label = layers[1]
    assert isinstance(first_label, TextLayer) and first_label.text == f"{levels[0]:.2g}"


def test_indifference_explicit_levels_highlight_and_ordinal_labels():
    func = CobbDouglas(0.5, 0.5)
    layers, levels = indifference_layers(
        func, [2.0, 4.0, 6.0], 15, 10, highlight_level=4.1, show_labels=True, label_style="ordinal", legend="IC"
    )
    assert levels == [2.0, 4.0, 6.0]
    paths = {layer.id: layer for layer in layers if isinstance(layer, PathLayer)}
    assert paths["ic.2"].role == Indifference.MAIN and paths["ic.2"].legend == "IC"
    assert paths["ic.1"].role == Indifference.SECONDARY and paths["ic.3"].role == Indifference.SECONDARY
    texts = {layer.id: layer for layer in layers if isinstance(layer, TextLayer)}
    assert texts["ic.1.label"].text == "u_{1}" and texts["ic.1.label"].math
    assert texts["ic.1.label"].role == Indifference.SECONDARY_LABEL


def test_level_outside_the_box_is_skipped():
    layers, levels = indifference_layers(CobbDouglas(), [1e9], 10, 10)
    assert layers == () and levels == [1e9]

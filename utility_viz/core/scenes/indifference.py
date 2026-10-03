"""Indifference curves as backend-neutral mosaickit layers.

Contour *level selection* (``percentile_levels``) and the curve tracing both live in
``utility_viz.models.curves``; this module only wraps the resulting polylines in layers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from mosaickit import Layer, PathLayer, Stroke, TextLayer

from utility_viz.core.constants.canvas import CONTOUR_DOMAIN_MIN
from utility_viz.core.scenes.roles import Indifference
from utility_viz.models.curves import Layer as ContourGrid
from utility_viz.models.curves import level_curve_path, percentile_levels


def indifference_layers(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    levels: int | Sequence[float],
    x_max: float,
    y_max: float,
    *,
    res: int = 400,
    layer_id: str = "ic",
    highlight_level: float | None = None,
    show_labels: bool = False,
    label_fmt: str = "{:.2g}",
    label_style: str = "numeric",
    stroke: Stroke | None = None,
    legend: str | None = None,
    z_index: float = 0,
) -> tuple[tuple[Layer, ...], list[float]]:
    """Layers for the indifference curves of *func* over ``[0, x_max] x [0, y_max]``.

    *levels* is a count (auto-spaced by ``percentile_levels``) or explicit utility
    values. Returns ``(layers, computed_levels)``. Each level becomes a
    :class:`~mosaickit.PathLayer` with id ``<layer_id>.<rank>`` (rank from 1, ascending);
    with *show_labels*, a ``<layer_id>.<rank>.label`` :class:`~mosaickit.TextLayer` sits at
    the curve's right end. When *highlight_level* is given only the nearest level uses
    the ``utility.indifference`` role, the rest the subdued ``.secondary`` role.
    *legend* names the first (focal) curve only. Curve tracing assumes a utility that is
    non-decreasing in both goods (see :func:`~utility_viz.models.curves.level_curve_path`).
    """
    domain = (CONTOUR_DOMAIN_MIN, x_max), (CONTOUR_DOMAIN_MIN, y_max)
    if isinstance(levels, int):
        _, _, z = ContourGrid.compute_contour(func, domain[0], domain[1], res=res)
        computed = percentile_levels(z, n=levels)
    else:
        computed = list(levels)

    focal: int | None = None
    if highlight_level is not None and computed:
        focal = int(np.argmin(np.abs(np.array(computed) - highlight_level)))

    layers: list[Layer] = []
    for index, level in enumerate(computed):
        path = level_curve_path(func, level, domain[0], domain[1], res=res)
        if not path:
            continue
        rank = index + 1
        is_focal = focal is None or index == focal
        main, label_role = (
            (Indifference.MAIN, Indifference.LABEL)
            if is_focal
            else (Indifference.SECONDARY, Indifference.SECONDARY_LABEL)
        )
        layers.append(
            PathLayer(
                path,
                stroke=stroke if is_focal else None,
                id=f"{layer_id}.{rank}",
                role=main.value,
                legend=legend if (legend and index == (focal or 0)) else None,
                z_index=z_index,
            )
        )
        if show_labels:
            anchor = _label_anchor(path, x_max, y_max)
            if anchor is not None:
                text = f"u_{{{rank}}}" if label_style == "ordinal" else label_fmt.format(level)
                layers.append(
                    TextLayer(
                        anchor,
                        text,
                        math=label_style == "ordinal",
                        offset=(4, 0),
                        anchor="left",
                        id=f"{layer_id}.{rank}.label",
                        role=label_role.value,
                        z_index=z_index + 1,
                    )
                )
    return tuple(layers), computed


def _label_anchor(path: list[tuple[float, float]], x_max: float, y_max: float) -> tuple[float, float] | None:
    """The rightmost point of *path* that stays inside 97% of the box (as the legacy labels)."""
    inside = [(x, y) for x, y in path if x < x_max * 0.97 and y < y_max * 0.97]
    return max(inside, default=None)

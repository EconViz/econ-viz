"""Budget constraint as backend-neutral mosaickit layers."""

from __future__ import annotations

from mosaickit import Fill, FillLayer, Layer, PathLayer, Stroke, TextLayer

from utility_viz.core.errors.exceptions import InvalidParameterError
from utility_viz.core.scenes.roles import Budget


def budget_layers(
    px: float,
    py: float,
    income: float,
    *,
    layer_id: str = "budget",
    role: str = Budget.MAIN,
    fill_role: str = Budget.FILL,
    fill: bool = False,
    label: str | None = None,
    stroke: Stroke | None = None,
    fill_style: Fill | None = None,
    legend: str | None = None,
    z_index: float = 1,
) -> tuple[Layer, ...]:
    """Layers for the line ``px*x + py*y = income``.

    Returns the budget :class:`~mosaickit.PathLayer` (id ``layer_id``), preceded by a
    ``<layer_id>.fill`` :class:`~mosaickit.FillLayer` of the feasible set when *fill* is
    set, and followed by a ``<layer_id>.label`` :class:`~mosaickit.TextLayer` (LaTeX
    math, written without ``$``) near the line's midpoint when *label* is given. Styling
    comes from the theme roles; *stroke* and *fill_style* are sparse per-layer overrides.
    """
    if px <= 0 or py <= 0 or income <= 0:
        raise InvalidParameterError(f"Budget parameters must be positive (px={px}, py={py}, income={income}).")
    x_int, y_int = income / px, income / py

    layers: list[Layer] = []
    if fill:
        layers.append(
            FillLayer(
                [(0.0, 0.0), (x_int, 0.0), (0.0, y_int)],
                fill=fill_style,
                id=f"{layer_id}.fill",
                role=str(getattr(fill_role, "value", fill_role)),
                z_index=z_index - 1,
            )
        )
    role_name = str(getattr(role, "value", role))
    layers.append(
        PathLayer(
            [(x_int, 0.0), (0.0, y_int)],
            stroke=stroke,
            id=layer_id,
            role=role_name,
            legend=legend,
            z_index=z_index,
        )
    )
    if label:
        layers.append(
            TextLayer(
                (x_int / 2, y_int / 2),
                label,
                math=True,
                offset=(6, 6),
                anchor="bottom-left",
                id=f"{layer_id}.label",
                role=role_name,
                z_index=z_index + 1,
            )
        )
    return tuple(layers)

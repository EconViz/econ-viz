"""Equilibrium bundle as backend-neutral mosaickit layers."""

from __future__ import annotations

from mosaickit import Layer, Marker, MarkerLayer, PathLayer, Stroke, TextLayer

from utility_viz.core.scenes.roles import Equilibrium
from utility_viz.models.optimization.solver import Equilibrium as EquilibriumResult


def equilibrium_layers(
    eq: EquilibriumResult,
    *,
    layer_id: str = "equilibrium",
    role: str = Equilibrium.MAIN,
    label: str | None = "x^*",
    drop_lines: bool = True,
    ray_to: tuple[float, float] | None = None,
    marker: Marker | None = None,
    drop_stroke: Stroke | None = None,
    ray_stroke: Stroke | None = None,
    legend: str | None = None,
    z_index: float = 6,
) -> tuple[Layer, ...]:
    """Layers for a solved equilibrium bundle ``eq``.

    * ``<layer_id>`` -- :class:`~mosaickit.MarkerLayer` at ``(eq.x, eq.y)``, carrying *eq*
      as the layer ``model``.
    * ``<layer_id>.label`` -- :class:`~mosaickit.TextLayer` (LaTeX math without ``$``) when
      *label* is set.
    * ``<layer_id>.drop`` -- one :class:`~mosaickit.PathLayer` running axis -> bundle -> axis
      when *drop_lines* is set.
    * ``<layer_id>.ray`` -- the expansion-path ray from the origin through the bundle, up to
      the ``(x_max, y_max)`` box given as *ray_to*; omitted when *ray_to* is ``None``.
    """
    role_name = str(getattr(role, "value", role))
    layers: list[Layer] = [
        MarkerLayer(
            [(eq.x, eq.y)],
            marker=marker,
            id=layer_id,
            role=role_name,
            legend=legend,
            model=eq,
            z_index=z_index,
        )
    ]
    if label:
        layers.append(
            TextLayer(
                (eq.x, eq.y),
                label,
                math=True,
                offset=(5, 5),
                anchor="bottom-left",
                id=f"{layer_id}.label",
                role=role_name,
                z_index=z_index + 1,
            )
        )
    if drop_lines:
        layers.append(
            PathLayer(
                [(0.0, eq.y), (eq.x, eq.y), (eq.x, 0.0)],
                stroke=drop_stroke,
                id=f"{layer_id}.drop",
                role=Equilibrium.DROP.value,
                z_index=z_index - 4,
            )
        )
    if ray_to is not None and eq.x > 1e-9 and eq.y > 1e-9:
        scale = min(ray_to[0] / eq.x, ray_to[1] / eq.y)
        layers.append(
            PathLayer(
                [(0.0, 0.0), (eq.x * scale, eq.y * scale)],
                stroke=ray_stroke,
                id=f"{layer_id}.ray",
                role=Equilibrium.RAY.value,
                z_index=z_index - 5,
            )
        )
    return tuple(layers)

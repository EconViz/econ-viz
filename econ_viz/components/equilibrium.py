"""Equilibrium point component."""

from __future__ import annotations

from ..logging import get_logger

logger = get_logger(__name__)


class EquilibriumPoint:
    """Renders an equilibrium bundle annotation on the canvas.

    Parameters
    ----------
    eq : Equilibrium
        A pre-solved equilibrium bundle.
    color : str
        Colour for the marker and drop-lines.
    markersize : float
        Size of the equilibrium dot.
    label : str or None
        LaTeX label placed next to the dot.
    drop_dashes : bool
        Draw dashed perpendicular lines from the optimum to both axes.
    show_ray : bool
        Draw the expansion-path ray from the origin through the optimum.
    ray_color : str
        Colour for the expansion-path ray.
    ray_linewidth : float
        Stroke width for the expansion-path ray.
    """

    def __init__(
        self,
        eq,
        color: str,
        markersize: float = 6.0,
        label: str | None = "x^*",
        drop_dashes: bool = True,
        show_ray: bool = False,
        ray_color: str | None = None,
        ray_linewidth: float = 0.8,
    ):
        self.eq = eq
        self.color = color
        self.markersize = markersize
        self.label = label
        self.drop_dashes = drop_dashes
        self.show_ray = show_ray
        self.ray_color = ray_color or color
        self.ray_linewidth = ray_linewidth

    def draw(self, ax, x_max: float, y_max: float) -> None:
        """Draw the equilibrium annotation onto *ax*."""
        from . import draw_ray

        eq = self.eq
        logger.info(
            "Equilibrium (%s): x=%.4f, y=%.4f, U=%.4f",
            eq.bundle_type, eq.x, eq.y, eq.utility,
        )

        ax.plot(eq.x, eq.y, "o", color=self.color, markersize=self.markersize)
        if self.label:
            ax.annotate(
                rf"${self.label}$", (eq.x, eq.y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=12, color=self.color,
            )

        if self.drop_dashes:
            dash_kw = dict(color=self.color, linestyle=":", linewidth=0.8)
            ax.plot([eq.x, eq.x], [0, eq.y], **dash_kw)
            ax.plot([0, eq.x], [eq.y, eq.y], **dash_kw)

        if self.show_ray and eq.x > 1e-9:
            draw_ray(ax, eq.y / eq.x, x_max, y_max,
                     color=self.ray_color, linewidth=self.ray_linewidth)

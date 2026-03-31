"""Theme data structure for controlling diagram appearance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Colour and style specification for economic diagrams.

    Every visual element reads its default appearance from the active
    theme.  Per-call overrides are still possible via method arguments
    on :class:`~econ_viz.canvas.base.Canvas`.

    Parameters
    ----------
    name : str
        Human-readable theme identifier.
    axis_color : str
        Colour for axis spines, ticks, and arrow terminators.
    label_color : str
        Colour for axis labels and the origin marker.
    ic_color : str
        Default colour for indifference curves.
    ic_linewidth : float
        Default stroke width for indifference curves.
    path_color : str
        Default colour for PCC / ICC path lines.
    path_linewidth : float
        Default stroke width for PCC / ICC path lines.
    budget_color : str
        Default colour for budget constraint lines.
    budget_linewidth : float
        Default stroke width for budget constraint lines.
    budget_fill_alpha : float
        Opacity of the feasible-set shading.
    eq_color : str
        Default colour for equilibrium markers and drop-lines.
    eq_markersize : float
        Default marker size for equilibrium points.
    ray_color : str
        Default colour for expansion-path / kink-locus rays.
    ray_linewidth : float
        Default stroke width for rays.
    kink_color : str
        Default colour for kink-point markers.
    """

    name: str

    # Axes & labels
    axis_color: str = "black"
    label_color: str = "black"

    # Indifference curves
    ic_color: str = "black"
    ic_linewidth: float = 2.0

    # PCC / ICC paths
    path_color: str = "#2A9D8F"
    path_linewidth: float = 2.0

    # Budget constraint
    budget_color: str = "blue"
    budget_linewidth: float = 1.5
    budget_fill_alpha: float = 0.08

    # Equilibrium
    eq_color: str = "red"
    eq_markersize: float = 6.0

    # Rays
    ray_color: str = "black"
    ray_linewidth: float = 0.8

    # Kink markers
    kink_color: str = "black"

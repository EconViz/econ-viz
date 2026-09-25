"""Theme data structure for controlling diagram appearance."""

from dataclasses import dataclass

from ..enums import ArrowStyle, LineStyle
from .stroke import Stroke


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
    sub_effect_color : str
        Default colour for substitution-effect arrows/labels.
    inc_effect_color : str
        Default colour for income-effect arrows/labels.
    effect_arrow_linewidth : float
        Default stroke width for decomposition arrows.
    compensated_budget_color : str
        Default colour for compensated budget lines.
    compensated_budget_linewidth : float
        Default stroke width for compensated budget lines.
    compensated_budget_linestyle : str
        Default line style for compensated budget lines.
    axis_stroke : Stroke
        Axis lines and their arrowheads. Colour ``None`` uses *axis_color*.
    drop_stroke : Stroke
        Dashed guides from an equilibrium point to the axes. Colour ``None``
        uses *eq_color*.
    projection_stroke : Stroke
        Vertical guides from decomposition bundles down to the x-axis.
    guide_stroke : Stroke
        Guides below the x-axis in decomposition diagrams.
    box_stroke : Stroke
        Frame of an Edgeworth box. Colour ``None`` uses *axis_color*.

    Default strokes
    ---------------
    Every line has a default :class:`Stroke`; a Stroke passed to a Canvas
    method only overrides the fields it sets.

    ========================== =========================================
    Line                       Default
    ========================== =========================================
    axes                       ``axis_stroke``: 0.8 pt, solid, ``-|>``
    indifference curves        ``ic_stroke``: *ic_linewidth*, solid
    budget line                ``budget_stroke``: *budget_linewidth*, solid
    rays                       ``ray_stroke``: *ray_linewidth*, dashed
    equilibrium drop lines     ``drop_stroke``: 0.8 pt, dotted
    PCC / ICC paths            ``path_stroke``: *path_linewidth*, solid
    compensated budget         ``compensated_budget_stroke``
    final budget               ``final_budget_stroke``: dashdot
    substitution / income      ``substitution_stroke`` / ``income_stroke``:
                               *effect_arrow_linewidth*, solid, ``->``
    decomposition projections  ``projection_stroke``: 0.8 pt, dotted
    guides below the x-axis    ``guide_stroke``: 0.8 pt, dashed
    Edgeworth box frame        ``box_stroke``: 1.2 pt, solid
    ========================== =========================================
    """

    name: str

    # Axes & labels
    axis_color: str = "#222222"
    label_color: str = "#222222"

    # Indifference curves
    ic_color: str = "#377EB8"
    ic_linewidth: float = 1.8

    # PCC / ICC paths
    path_color: str = "#4DAF4A"
    path_linewidth: float = 2.0

    # Budget constraint
    budget_color: str = "#984EA3"
    budget_linewidth: float = 1.5
    budget_fill_alpha: float = 0.08

    # Equilibrium
    eq_color: str = "#E41A1C"
    eq_markersize: float = 6.0

    # Rays
    ray_color: str = "#999999"
    ray_linewidth: float = 0.8

    # Kink markers
    kink_color: str = "#A65628"

    # Price decomposition visuals
    sub_effect_color: str = "#FF7F00"
    inc_effect_color: str = "#4DAF4A"
    effect_arrow_linewidth: float = 1.6
    compensated_budget_color: str = "#777777"
    compensated_budget_linewidth: float = 1.5
    compensated_budget_linestyle: str = "--"
    # Lines without their own width/colour fields
    axis_stroke: Stroke = Stroke(width=0.8, style=LineStyle.SOLID, arrow=ArrowStyle.TRIANGLE)
    drop_stroke: Stroke = Stroke(width=0.8, style=LineStyle.DOTTED)
    projection_stroke: Stroke = Stroke(width=0.8, style=LineStyle.DOTTED, color="#888888")
    guide_stroke: Stroke = Stroke(width=0.8, style=LineStyle.DASHED, color="#777777")
    box_stroke: Stroke = Stroke(width=1.2, style=LineStyle.SOLID)

    @property
    def ic_stroke(self) -> Stroke:
        return Stroke(width=self.ic_linewidth, style=LineStyle.SOLID, color=self.ic_color)

    @property
    def budget_stroke(self) -> Stroke:
        return Stroke(width=self.budget_linewidth, style=LineStyle.SOLID, color=self.budget_color)

    @property
    def ray_stroke(self) -> Stroke:
        return Stroke(width=self.ray_linewidth, style=LineStyle.DASHED, color=self.ray_color)

    @property
    def path_stroke(self) -> Stroke:
        return Stroke(width=self.path_linewidth, style=LineStyle.SOLID, color=self.path_color)

    @property
    def compensated_budget_stroke(self) -> Stroke:
        return Stroke(
            width=self.compensated_budget_linewidth,
            style=self.compensated_budget_linestyle,
            color=self.compensated_budget_color,
        )

    @property
    def final_budget_stroke(self) -> Stroke:
        return Stroke(width=self.budget_linewidth, style=LineStyle.DASHDOT, color=self.budget_color)

    @property
    def substitution_stroke(self) -> Stroke:
        return Stroke(
            width=self.effect_arrow_linewidth,
            style=LineStyle.SOLID,
            color=self.sub_effect_color,
            arrow=ArrowStyle.SIMPLE,
        )

    @property
    def income_stroke(self) -> Stroke:
        return Stroke(
            width=self.effect_arrow_linewidth,
            style=LineStyle.SOLID,
            color=self.inc_effect_color,
            arrow=ArrowStyle.SIMPLE,
        )

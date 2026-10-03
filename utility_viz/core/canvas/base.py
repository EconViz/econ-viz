"""Canvas — the central plotting surface for economic diagrams.

A :class:`Canvas` instance manages a single matplotlib figure styled in the
convention of microeconomic textbook diagrams: first-quadrant axes with
LaTeX-rendered labels at the axis tips, origin marker, arrow terminators,
and no numeric tick labels.

Drawing is delegated to component classes in :mod:`utility_viz.core.diagrams.components`;
the canvas itself is a thin orchestration layer that resolves theme
defaults and forwards calls. Its behaviour is split across sibling modules:
``style`` (base styling), ``layers`` (curves, budgets, points, paths),
``decomposition``, ``legend`` and ``output``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt

from utility_viz.core.canvas._labels import _axis_stroke, _label_position
from utility_viz.core.canvas.decomposition import DecompositionMixin
from utility_viz.core.canvas.output import OutputMixin
from utility_viz.core.canvas.style import apply_base_style
from utility_viz.core.config.settings import Config
from utility_viz.core.constants.canvas import DEFAULT_DPI, MAX_DPI, MIN_DPI
from utility_viz.core.rendering.fonts import FontApplier, resolve_font, resolve_math_font
from utility_viz.core.styles.axis import Axis
from utility_viz.core.styles.label import Label, split_label
from utility_viz.core.styles.stroke import Stroke
from utility_viz.core.themes.theme import Theme
from utility_viz.enums import ArrowStyle, LabelPosition, LineStyle
from utility_viz.utils.logging import get_logger

logger = get_logger(__name__)


class Canvas(DecompositionMixin, OutputMixin):
    """First-quadrant plotting surface for economic visualizations.

    Parameters
    ----------
    x_max : float
        Upper bound of the horizontal axis.
    y_max : float
        Upper bound of the vertical axis.
    x_label : str
        Label at the tip of the horizontal axis (rendered in LaTeX math mode).
    y_label : str
        Label at the tip of the vertical axis (rendered in LaTeX math mode).
    title : str or None
        Optional figure title.
    dpi : int
        Resolution for raster export. Clamped to ``[1, 1200]``. Default 300.
    x_label_pos : LabelPosition or str
        Position around the x-axis arrowhead: top, right, or bottom.
    y_label_pos : LabelPosition or str
        Position around the y-axis arrowhead: left, top, or right.
    theme : Theme, optional
        Colour and style theme. Defaults to the active :class:`Config`'s theme
        (the built-in ``default`` theme unless ``Config.use()`` was called).
    x_arrow_style, y_arrow_style : ArrowStyle or str
        Independently configurable arrowhead styles for each axis.
    font : str or sequence of str, optional
        Font family for every text element on this canvas, or a fallback list.
        Generic families (``"serif"``, ``"sans-serif"``, ``"monospace"``) are
        accepted. ``None`` keeps Matplotlib's default. Global rcParams are not
        modified.
    math_font : str, optional
        Matplotlib math font set for math text such as axis labels:
        ``"dejavusans"``, ``"dejavuserif"``, ``"cm"``, ``"stix"``, or
        ``"stixsans"``. ``None`` keeps Matplotlib's default.
    x_line_style, y_line_style : LineStyle or str
        Line style of each axis: solid, dashed, dotted, or dashdot.
    axis_stroke : Stroke, optional
        Width, line style, colour, and arrowhead for both axes
        (default ``theme.axis_stroke``).
    x_axis_stroke, y_axis_stroke : Stroke, optional
        Per-axis overrides on top of *axis_stroke*.
    x_axis, y_axis : Axis, optional
        Label, label position, and stroke of one axis in a single object. The
        ``x_label`` / ``x_label_pos`` / ``x_*_style`` / ``*_axis_stroke``
        arguments are shorthand for it. Stroke precedence, highest first:
        ``Axis.stroke``, ``x_axis_stroke``, ``axis_stroke``,
        ``x_line_style`` / ``x_arrow_style``, ``theme.axis_stroke``.
        ``Axis.label`` may be a :class:`Label` for font size, colour, distance,
        and visibility (default ``theme.axis_label``).
    origin_label : str or Label, optional
        The ``0`` at the origin (default ``theme.origin_label``);
        ``Label(visible=False)`` hides it.

    *title* may also be a :class:`Label` for its font size and colour
    (default ``theme.title_label``).
    """

    def __init__(
        self,
        x_max: float = 10.0,
        y_max: float = 10.0,
        x_label: str = "X",
        y_label: str = "Y",
        title: str | Label | None = None,
        dpi: int = DEFAULT_DPI,
        x_label_pos: LabelPosition | str = LabelPosition.RIGHT,
        y_label_pos: LabelPosition | str = LabelPosition.TOP,
        theme: Theme | None = None,
        fig=None,
        ax=None,
        x_arrow_style: ArrowStyle | str | None = None,
        y_arrow_style: ArrowStyle | str | None = None,
        font: str | Sequence[str] | None = None,
        math_font: str | None = None,
        x_line_style: LineStyle | str | None = None,
        y_line_style: LineStyle | str | None = None,
        axis_stroke: Stroke | None = None,
        x_axis_stroke: Stroke | None = None,
        y_axis_stroke: Stroke | None = None,
        x_axis: Axis | None = None,
        y_axis: Axis | None = None,
        origin_label: str | Label | None = None,
    ):
        active = Config.active()
        theme = theme if theme is not None else active.theme
        font = font if font is not None else active.font
        math_font = math_font if math_font is not None else active.math_font
        x_axis, y_axis = x_axis or Axis(), y_axis or Axis()
        x_text, self.x_label_style = split_label(x_axis.label, theme.axis_label)
        y_text, self.y_label_style = split_label(y_axis.label, theme.axis_label)
        self.title, self.title_style = split_label(title, theme.title_label)
        origin_text, self.origin_style = split_label(origin_label, theme.origin_label, theme.origin_label.text)
        self.origin_text = origin_text or theme.origin_label.text or "0"
        self.x_max = x_max
        self.y_max = y_max
        self.x_label = x_text if x_text is not None else x_label
        self.y_label = y_text if y_text is not None else y_label
        self.dpi = max(MIN_DPI, min(dpi, MAX_DPI))
        self.x_label_pos = _label_position(
            x_axis.label_position or self.x_label_style.position or x_label_pos, axis="x"
        )
        self.y_label_pos = _label_position(
            y_axis.label_position or self.y_label_style.position or y_label_pos, axis="y"
        )
        self.theme = theme
        # Precedence: Axis.stroke > per-axis stroke > shared stroke > x/y_*_style arguments > theme.axis_stroke.
        self.x_axis_stroke = _axis_stroke(theme, x_line_style, x_arrow_style, axis_stroke, x_axis_stroke, x_axis.stroke)
        self.y_axis_stroke = _axis_stroke(theme, y_line_style, y_arrow_style, axis_stroke, y_axis_stroke, y_axis.stroke)
        self.x_line_style = self.x_axis_stroke.style
        self.y_line_style = self.y_axis_stroke.style
        self.x_arrow_style = self.x_axis_stroke.arrow
        self.y_arrow_style = self.y_axis_stroke.arrow
        self.font = resolve_font(font)
        self.math_font = resolve_math_font(math_font)

        self._owns_figure = fig is None or ax is None
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        else:
            self.fig, self.ax = fig, ax
        if (self.font or self.math_font) and self._owns_figure:
            self.fig.add_artist(FontApplier(self.font, self.math_font))
        self._legend_handles: list = []
        apply_base_style(self)
        logger.debug("Canvas created: x_max=%s, y_max=%s, dpi=%s, theme=%s", x_max, y_max, self.dpi, theme.name)

    # ------------------------------------------------------------------
    # Base styling
    # ------------------------------------------------------------------

    def _apply_base_style(self) -> None:
        """Configure axes to match textbook economic diagram conventions."""
        apply_base_style(self)

    def set_axis_visibility(self, *, show_x_label: bool = True, show_y_label: bool = True) -> Canvas:
        """Toggle canvas axis-tip labels and origin marker for shared layouts."""
        if not show_x_label:
            for text in list(self.ax.texts):
                if getattr(text, "_ev_axis_label", None) == "x":
                    text.set_visible(False)
        if not show_y_label:
            for text in list(self.ax.texts):
                if getattr(text, "_ev_axis_label", None) == "y":
                    text.set_visible(False)
        return self

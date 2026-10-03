"""Edgeworth box diagram for two-consumer exchange economies."""

from __future__ import annotations

import matplotlib.pyplot as plt

from utility_viz.core.canvas.legend import place_legend
from utility_viz.core.config.settings import Config
from utility_viz.core.constants.canvas import DEFAULT_DPI, MAX_DPI, MIN_DPI
from utility_viz.core.diagrams.consumer.edgeworth.methods import IndifferenceMixin
from utility_viz.core.export import save_figure
from utility_viz.core.styles.axis import Axis
from utility_viz.core.styles.label import Label
from utility_viz.core.styles.legend import Legend
from utility_viz.core.styles.stroke import Stroke
from utility_viz.core.themes.theme import Theme
from utility_viz.models.consumer.edgeworth_state import EdgeworthState


class EdgeworthBox(IndifferenceMixin):
    """Render a two-consumer Edgeworth box with exchange-theory primitives."""

    def __init__(
        self,
        utility_a,
        utility_b,
        total_x: float,
        total_y: float,
        *,
        x_label: str = "x",
        y_label: str = "y",
        title: str | Label | None = None,
        dpi: int = DEFAULT_DPI,
        theme: Theme | None = None,
        utility_a_color: str | None = None,
        utility_b_color: str | None = None,
        box_stroke: Stroke | None = None,
        x_axis: Axis | None = None,
        y_axis: Axis | None = None,
        origin_label: Label | None = None,
    ):
        """*x_axis* / *y_axis* (:class:`Axis`) name a good (drawn as ``x_A``,
        ``x_B``) and restyle the box sides along it: bottom and top for x,
        left and right for y, over *box_stroke*. The box has fixed label
        places, so ``label_position`` is not supported. ``Axis.label`` may be
        a :class:`Label` for the good names' font size, colour, and
        visibility; *origin_label* does the same for ``O_A`` / ``O_B`` and
        *title* may be a Label too (defaults ``theme.box_label`` /
        ``theme.title_label``).
        """
        if total_x <= 0 or total_y <= 0:
            raise ValueError("total_x and total_y must be positive.")
        theme = theme if theme is not None else Config.active().theme
        x_axis, y_axis = x_axis or Axis(), y_axis or Axis()
        self._setup_labels(theme, x_axis, y_axis, x_label, y_label, origin_label, title)

        self.utility_a = utility_a
        self.utility_b = utility_b
        self.total_x = float(total_x)
        self.total_y = float(total_y)
        self.dpi = max(MIN_DPI, min(int(dpi), MAX_DPI))
        self.theme = theme
        self._setup_strokes(theme, box_stroke, x_axis, y_axis)
        self.utility_a_color = utility_a_color or theme.ic_color
        self.utility_b_color = utility_b_color or theme.path_color

        self._state = EdgeworthState()

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self._apply_base_style()

    def set_utility_colors(self, *, color_a: str, color_b: str) -> EdgeworthBox:
        """Update default colors for utility A/B curves."""
        self.utility_a_color = color_a
        self.utility_b_color = color_b
        return self

    def show_legend(self, legend: Legend | None = None, **kwargs) -> EdgeworthBox:
        """Draw the legend.

        *legend* sets position, font size, frame, and columns (default
        ``theme.legend`` at 10 pt, placed where it covers the least of the
        box). Matplotlib ``**kwargs`` such as ``loc`` are used instead when given.
        """
        if kwargs:
            kwargs.setdefault("frameon", False)
            kwargs.setdefault("fontsize", 10)
            self.ax.legend(**kwargs)
            return self
        handles, labels = self.ax.get_legend_handles_labels()
        place_legend(
            self.ax,
            handles,
            labels,
            (legend or Legend()).merged_over(Legend(fontsize=10).merged_over(self.theme.legend)),
        )
        return self

    def save(self, path: str, **kwargs) -> None:
        """Export the Edgeworth box figure to disk."""
        kwargs.setdefault("transparent", self.theme.background_color is None)
        save_figure(self.fig, path=path, dpi=self.dpi, close=True, **kwargs)

    def show(self) -> None:
        """Display the figure in an interactive matplotlib window."""
        self.fig.show()

"""Budget constraint component."""

from __future__ import annotations

from ..exceptions import InvalidParameterError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BudgetConstraint:
    """Renders a linear budget constraint px*x + py*y = income.

    Parameters
    ----------
    px, py : float
        Prices of goods *x* and *y*. Must be positive.
    income : float
        Total budget. Must be positive.
    color : str
        Line colour.
    linewidth : float
        Stroke width.
    linestyle : str
        Matplotlib line-style string.
    label : str or None
        Optional legend label (rendered in LaTeX math mode).
    fill : bool
        Shade the feasible set below the budget line.
    fill_alpha : float
        Opacity of the feasible-set shading.
    """

    def __init__(
        self,
        px: float,
        py: float,
        income: float,
        color: str,
        linewidth: float,
        linestyle: str = "-",
        label: str | None = None,
        fill: bool = False,
        fill_alpha: float = 0.08,
    ):
        if px <= 0 or py <= 0 or income <= 0:
            raise InvalidParameterError(
                f"Budget parameters must be positive (px={px}, py={py}, income={income})."
            )
        self.px = px
        self.py = py
        self.income = income
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.label = label
        self.fill = fill
        self.fill_alpha = fill_alpha

    def draw(self, ax) -> None:
        """Draw the budget line (and optional feasible-set fill) onto *ax*."""
        x_int = self.income / self.px
        y_int = self.income / self.py

        logger.debug(
            "Budget line: px=%.4f, py=%.4f, I=%.4f → x_int=%.4f, y_int=%.4f",
            self.px, self.py, self.income, x_int, y_int,
        )

        plot_label = rf"${self.label}$" if self.label else None
        ax.plot(
            [x_int, 0], [0, y_int],
            color=self.color, linewidth=self.linewidth,
            linestyle=self.linestyle, label=plot_label,
        )

        if self.fill:
            ax.fill_between(
                [0, x_int], [y_int, 0],
                alpha=self.fill_alpha, color=self.color,
            )

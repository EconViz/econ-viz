"""Contract curve, core and price line methods of the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from utility_viz.core.diagrams.consumer.edgeworth.base import BoxBase
from utility_viz.core.diagrams.consumer.edgeworth.exchange import core_allocations
from utility_viz.core.diagrams.consumer.edgeworth.plotting import plot_contract_curve, plot_core, plot_price_line
from utility_viz.core.rendering.stroke import styled
from utility_viz.core.styles.marker import Marker
from utility_viz.core.styles.stroke import Stroke
from utility_viz.enums import LineStyle

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


class LinesMixin(BoxBase):
    """``add_contract_curve``, ``add_core`` and ``add_price_line``."""

    def add_contract_curve(
        self,
        *,
        n: int = 120,
        color: str | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        tolerance: float = 0.05,
        method: str = "auto",
        stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Approximate and draw the contract curve.

        Prefer *stroke*; *color*, *linewidth*, and *linestyle* are shorthand
        that a :class:`Stroke` supersedes.

        Parameters
        ----------
        color : str, optional
            Line colour. *None* → ``theme.contract_color``.
        linewidth : float, optional
            Line width. *None* → ``theme.contract_linewidth``.
        linestyle : str, optional
            Matplotlib line-style string. *None* → ``theme.contract_stroke``'s style.
        stroke : Stroke, optional
            Line style for the contract curve (default ``theme.contract_stroke``).
        """
        t = self.theme
        with styled(self, {"contract": stroke}):
            if method not in {"auto", "mrs", "pareto"}:
                raise ValueError("method must be one of: auto, mrs, pareto.")

            points = np.empty((0, 2), dtype=float)
            if method in {"auto", "mrs"}:
                points = self._contract_curve_mrs(n=n, tolerance=tolerance)
            if len(points) < 4 and method in {"auto", "pareto"}:
                points = self._contract_curve_pareto(n=n)

            self.contract_curve_points = points
            # Theme.contract_stroke always normalises style to a LineStyle.
            plot_contract_curve(
                self.ax,
                points=points,
                color=color or t.contract_color,
                linewidth=linewidth if linewidth is not None else t.contract_linewidth,
                linestyle=linestyle or cast(LineStyle, t.contract_stroke.style).value,
                label="Contract curve",
            )
        return self._as_box()

    def add_core(
        self,
        *,
        color: str | None = None,
        linewidth: float | None = None,
        min_points: int = 2,
        tol: float = 1e-6,
        stroke: Stroke | None = None,
        marker: Marker | None = None,
    ) -> EdgeworthBox:
        """Draw the core segment (IR part of the contract curve).

        Prefer *stroke*; *color* and *linewidth* are shorthand that a
        :class:`Stroke` supersedes.

        Parameters
        ----------
        color : str, optional
            Line/point colour. *None* → ``theme.core_color``.
        linewidth : float, optional
            Line width. *None* → ``theme.core_linewidth``.
        stroke : Stroke, optional
            Line style for the core (default ``theme.core_stroke``).
        marker : Marker, optional
            Colour and shape when the core collapses to a single point
            (default ``theme.core_marker``).
        """
        t = self.theme
        with styled(self, {"core": stroke}, markers={"core_point": marker}):
            core = core_allocations(self._as_box(), tol)

            self.core_points = self._unique_points(core)
            plot_core(
                self.ax,
                core_points=self.core_points,
                color=color or t.core_color,
                linewidth=linewidth if linewidth is not None else t.core_linewidth,
                label="Core",
                min_points=min_points,
            )
        return self._as_box()

    def add_price_line(
        self,
        px: float,
        py: float,
        *,
        color: str | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        label: str = "Price line",
        stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Draw the price line through endowment with slope -px/py.

        Prefer *stroke*; *color*, *linewidth*, and *linestyle* are shorthand
        that a :class:`Stroke` supersedes.

        Parameters
        ----------
        color : str, optional
            Line colour. *None* → ``theme.price_color``.
        linewidth : float, optional
            Line width. *None* → ``theme.price_linewidth``.
        linestyle : str, optional
            Matplotlib line-style string. *None* → ``theme.price_stroke``'s style.
        stroke : Stroke, optional
            Line style for the price line (default ``theme.price_stroke``).
        """
        t = self.theme
        with styled(self, {"price": stroke}):
            if px <= 0 or py <= 0:
                raise ValueError("px and py must be positive.")
            if self.endowment is None:
                raise ValueError("Endowment is required. Call add_endowment(...) first.")

            ex, ey = self.endowment
            income = px * ex + py * ey
            pts = self._line_box_intersections(px, py, income)
            plot_price_line(
                self.ax,
                points=pts,
                color=color or t.price_color,
                linewidth=linewidth if linewidth is not None else t.price_linewidth,
                # Theme.price_stroke always normalises style to a LineStyle.
                linestyle=linestyle or cast(LineStyle, t.price_stroke.style).value,
                label=label,
            )
        return self._as_box()

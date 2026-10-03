"""Indifference-map methods of the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utility_viz.core.diagrams.consumer.edgeworth.config import EquilibriumFocusConfig
from utility_viz.core.diagrams.consumer.edgeworth.focus import apply_equilibrium_focus, should_include_endowment_ic
from utility_viz.core.diagrams.consumer.edgeworth.methods.points import PointsMixin
from utility_viz.core.diagrams.consumer.edgeworth.plotting import plot_indifference_pair
from utility_viz.core.rendering.stroke import styled
from utility_viz.core.styles.stroke import Stroke
from utility_viz.models.curves import percentile_levels

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


class IndifferenceMixin(PointsMixin):
    """Indifference curves, their equilibrium variants and the equilibrium-focus view."""

    def _utility_surfaces(self, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Grid ``X, Y`` with A's and B's utility on it (B measured from the far corner)."""
        X, Y = self._grid(res=res)
        return X, Y, self.utility_a(X, Y), self.utility_b(self.total_x - X, self.total_y - Y)

    def add_indifference_curves(
        self,
        *,
        levels_a: int | list[float] = 4,
        levels_b: int | list[float] = 4,
        color_a: str | None = None,
        color_b: str | None = None,
        linewidth: float | None = None,
        res: int = 320,
        stroke_a: Stroke | None = None,
        stroke_b: Stroke | None = None,
    ) -> EdgeworthBox:
        """Draw both consumers' indifference maps.

        Prefer *stroke_a* / *stroke_b*; *color_a*, *color_b*, and *linewidth*
        are shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
        stroke_a : Stroke, optional
            Line style for consumer A's indifference curves.
        stroke_b : Stroke, optional
            Line style for consumer B's indifference curves.
        """
        with styled(self, {"curve_a": stroke_a, "curve_b": stroke_b}):
            t = self.theme
            lw = linewidth if linewidth is not None else t.ic_linewidth
            ca = color_a or self.utility_a_color
            cb = color_b or self.utility_b_color

            X, Y, U_a, U_b = self._utility_surfaces(res)
            lv_a = percentile_levels(U_a, n=levels_a) if isinstance(levels_a, int) else list(levels_a)
            lv_b = percentile_levels(U_b, n=levels_b) if isinstance(levels_b, int) else list(levels_b)
            plot_indifference_pair(
                self.ax,
                X=X,
                Y=Y,
                U_a=U_a,
                U_b=U_b,
                levels_a=lv_a,
                levels_b=lv_b,
                color_a=ca,
                color_b=cb,
                linewidth=lw,
            )
        return self._as_box()

    def add_endowment_indifference(
        self,
        *,
        color_a: str | None = None,
        color_b: str | None = None,
        linewidth: float | None = None,
        res: int = 300,
        stroke_a: Stroke | None = None,
        stroke_b: Stroke | None = None,
    ) -> EdgeworthBox:
        """Draw each agent's indifference curve through the endowment point.

        Prefer *stroke_a* / *stroke_b*; *color_a*, *color_b*, and *linewidth*
        are shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
        stroke_a : Stroke, optional
            Line style for consumer A's indifference curves.
        stroke_b : Stroke, optional
            Line style for consumer B's indifference curves.
        """
        with styled(self, {"curve_a": stroke_a, "curve_b": stroke_b}):
            if self.endowment is None:
                raise ValueError("Endowment is required. Call add_endowment(...) first.")

            ex, ey = self.endowment
            u_a_e = self._eval_ua(ex, ey)
            u_b_e = self._eval_ub(ex, ey)
            t = self.theme
            lw = linewidth if linewidth is not None else max(t.ic_linewidth, 1.8)
            ca = color_a or self.utility_a_color
            cb = color_b or self.utility_b_color

            X, Y, U_a, U_b = self._utility_surfaces(res)
            plot_indifference_pair(
                self.ax,
                X=X,
                Y=Y,
                U_a=U_a,
                U_b=U_b,
                levels_a=[u_a_e],
                levels_b=[u_b_e],
                color_a=ca,
                color_b=cb,
                linewidth=lw,
            )
        return self._as_box()

    def _equilibrium_utilities(self, px: float, py: float) -> tuple[float, float]:
        """Both agents' utility at ``X*``, locating the equilibrium first if needed."""
        if px <= 0 or py <= 0:
            raise ValueError("px and py must be positive.")
        if self.walrasian_equilibrium is None:
            self.add_walrasian_equilibrium(px=px, py=py)

        equilibrium = self.walrasian_equilibrium
        assert equilibrium is not None
        x_star, y_star = equilibrium
        return self._eval_ua(x_star, y_star), self._eval_ub(x_star, y_star)

    def add_indifference_curves_from_equilibrium(
        self,
        *,
        px: float,
        py: float,
        n_a: int = 4,
        n_b: int = 4,
        spread: float = 0.5,
        color_a: str | None = None,
        color_b: str | None = None,
        linewidth: float | None = None,
        res: int = 320,
        stroke_a: Stroke | None = None,
        stroke_b: Stroke | None = None,
        contract_stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Draw indifference curves around the Walrasian equilibrium utility levels.

        Prefer *stroke_a* / *stroke_b*; *color_a*, *color_b*, and *linewidth*
        are shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
        stroke_a : Stroke, optional
            Line style for consumer A's indifference curves.
        stroke_b : Stroke, optional
            Line style for consumer B's indifference curves.
        contract_stroke : Stroke, optional
            Line style for the contract curve.
        """
        with styled(self, {"curve_a": stroke_a, "curve_b": stroke_b, "contract": contract_stroke}):
            ua_star, ub_star = self._equilibrium_utilities(px, py)
            levels_a = self._levels_around(anchor=ua_star, n=n_a, spread=spread)
            levels_b = self._levels_around(anchor=ub_star, n=n_b, spread=spread)
        return self.add_indifference_curves(
            levels_a=levels_a,
            levels_b=levels_b,
            color_a=color_a,
            color_b=color_b,
            linewidth=linewidth,
            res=res,
            stroke_a=stroke_a,
            stroke_b=stroke_b,
        )

    def add_equilibrium_indifference(
        self,
        *,
        px: float,
        py: float,
        color_a: str | None = None,
        color_b: str | None = None,
        linewidth: float | None = None,
        res: int = 300,
        stroke_a: Stroke | None = None,
        stroke_b: Stroke | None = None,
        contract_stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Draw one indifference curve per agent through the Walrasian equilibrium.

        Prefer *stroke_a* / *stroke_b*; *color_a*, *color_b*, and *linewidth*
        are shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
        stroke_a : Stroke, optional
            Line style for consumer A's indifference curves.
        stroke_b : Stroke, optional
            Line style for consumer B's indifference curves.
        contract_stroke : Stroke, optional
            Line style for the contract curve.
        """
        with styled(self, {"curve_a": stroke_a, "curve_b": stroke_b, "contract": contract_stroke}):
            u_a_star, u_b_star = self._equilibrium_utilities(px, py)
        return self.add_indifference_curves(
            levels_a=[u_a_star],
            levels_b=[u_b_star],
            color_a=color_a,
            color_b=color_b,
            linewidth=linewidth,
            res=res,
            stroke_a=stroke_a,
            stroke_b=stroke_b,
        )

    def apply_equilibrium_focus(
        self,
        *,
        px: float,
        py: float,
        config: EquilibriumFocusConfig | None = None,
    ) -> EdgeworthBox:
        """Render only the most informative indifference curves around equilibrium.

        Draws a bounded number of ICs per agent around ``X*`` (default: 3-5).
        Endowment ICs are optional and, when included, compete for the same cap.
        """
        apply_equilibrium_focus(self._as_box(), px, py, config or EquilibriumFocusConfig())
        return self._as_box()

    def _should_include_endowment_ic(self, *, min_relative_gap: float) -> bool:
        return should_include_endowment_ic(self._as_box(), min_relative_gap=min_relative_gap)

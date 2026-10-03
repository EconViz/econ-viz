"""Edgeworth box diagram for two-consumer exchange economies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from utility_viz.core.canvas.legend import place_legend
from utility_viz.core.config.settings import Config
from utility_viz.core.constants.canvas import DEFAULT_DPI, MAX_DPI, MIN_DPI
from utility_viz.core.diagrams.consumer.edgeworth_exchange import (
    check_allocation,
    core_allocations,
    locate_walrasian,
)
from utility_viz.core.diagrams.consumer.edgeworth_focus import apply_equilibrium_focus, should_include_endowment_ic
from utility_viz.core.diagrams.consumer.edgeworth_plotter import (
    plot_contract_curve,
    plot_core,
    plot_endowment,
    plot_equilibrium_marker,
    plot_indifference_pair,
    plot_price_line,
)
from utility_viz.core.diagrams.consumer.edgeworth_style import apply_base_style
from utility_viz.core.errors.exceptions import InvalidParameterError
from utility_viz.core.export import save_figure
from utility_viz.core.rendering.stroke import styled
from utility_viz.core.styles.axis import Axis
from utility_viz.core.styles.label import Label, split_label
from utility_viz.core.styles.legend import Legend
from utility_viz.core.styles.marker import Marker
from utility_viz.core.styles.stroke import Stroke
from utility_viz.core.themes.theme import Theme
from utility_viz.enums import LineStyle
from utility_viz.models.consumer.edgeworth_compute import (
    contract_curve_mrs,
    contract_curve_pareto,
    focus_levels,
    line_box_intersections,
    mrs,
    unique_points,
)
from utility_viz.models.consumer.edgeworth_state import EdgeworthState
from utility_viz.models.curves import around_anchor_levels, percentile_levels

_EPS = 1e-3


@dataclass(frozen=True)
class EquilibriumFocusConfig:
    """Configuration for equilibrium-focused indifference rendering."""

    include_endowment_indifference: bool | str = "auto"
    min_relative_gap: float = 0.2
    min_curves_per_agent: int = 3
    max_curves_per_agent: int = 5
    equilibrium_spread: float = 0.35
    equilibrium_linewidth: float | None = None
    endowment_linewidth: float | None = None
    res: int = 300


class EdgeworthBox:
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
        for name, axis in (("x_axis", x_axis), ("y_axis", y_axis)):
            if axis.label_position is not None:
                raise InvalidParameterError(f"EdgeworthBox {name} does not support label_position")
        x_text, self.x_label_style = split_label(x_axis.label, theme.box_label)
        y_text, self.y_label_style = split_label(y_axis.label, theme.box_label)
        x_label = x_text if x_text is not None else x_label
        y_label = y_text if y_text is not None else y_label
        _, self.origin_style = split_label(origin_label, theme.box_label)
        title, self.title_style = split_label(title, theme.title_label)

        self.utility_a = utility_a
        self.utility_b = utility_b
        self.total_x = float(total_x)
        self.total_y = float(total_y)
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.dpi = max(MIN_DPI, min(int(dpi), MAX_DPI))
        self.theme = theme
        self.box_stroke = (
            (box_stroke or Stroke()).merged_over(theme.box_stroke).merged_over(Stroke(color=theme.axis_color))
        )
        self.x_side_stroke = (x_axis.stroke or Stroke()).merged_over(self.box_stroke)
        self.y_side_stroke = (y_axis.stroke or Stroke()).merged_over(self.box_stroke)
        self.utility_a_color = utility_a_color or theme.ic_color
        self.utility_b_color = utility_b_color or theme.path_color

        self._state = EdgeworthState()

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self._apply_base_style()

    @property
    def endowment(self) -> tuple[float, float] | None:
        return self._state.endowment

    @endowment.setter
    def endowment(self, value: tuple[float, float] | None) -> None:
        self._state.endowment = value

    @property
    def contract_curve_points(self) -> np.ndarray:
        return self._state.contract_curve_points

    @contract_curve_points.setter
    def contract_curve_points(self, value: np.ndarray) -> None:
        self._state.contract_curve_points = value

    @property
    def core_points(self) -> np.ndarray:
        return self._state.core_points

    @core_points.setter
    def core_points(self, value: np.ndarray) -> None:
        self._state.core_points = value

    @property
    def walrasian_equilibrium(self) -> tuple[float, float] | None:
        return self._state.walrasian_equilibrium

    @walrasian_equilibrium.setter
    def walrasian_equilibrium(self, value: tuple[float, float] | None) -> None:
        self._state.walrasian_equilibrium = value

    @property
    def equilibrium_focus_levels_a(self) -> list[float]:
        return self._state.equilibrium_focus_levels_a

    @equilibrium_focus_levels_a.setter
    def equilibrium_focus_levels_a(self, value: list[float]) -> None:
        self._state.equilibrium_focus_levels_a = value

    @property
    def equilibrium_focus_levels_b(self) -> list[float]:
        return self._state.equilibrium_focus_levels_b

    @equilibrium_focus_levels_b.setter
    def equilibrium_focus_levels_b(self, value: list[float]) -> None:
        self._state.equilibrium_focus_levels_b = value

    def set_utility_colors(self, *, color_a: str, color_b: str) -> EdgeworthBox:
        """Update default colors for utility A/B curves."""
        self.utility_a_color = color_a
        self.utility_b_color = color_b
        return self

    def _apply_base_style(self) -> None:
        apply_base_style(self)

    def _grid(self, *, res: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(_EPS, self.total_x - _EPS, res)
        y = np.linspace(_EPS, self.total_y - _EPS, res)
        return np.meshgrid(x, y)

    def _eval_ua(self, x: float, y: float) -> float:
        return float(self.utility_a(x, y))

    def _eval_ub(self, x: float, y: float) -> float:
        return float(self.utility_b(self.total_x - x, self.total_y - y))

    def _mrs(self, func, x: float, y: float, h: float = 1e-4) -> float:
        return mrs(func, x, y, x_max=self.total_x, y_max=self.total_y, eps=_EPS, h=h)

    def _unique_points(self, points: list[tuple[float, float]], digits: int = 4) -> np.ndarray:
        return unique_points(points, digits=digits)

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

            X, Y = self._grid(res=res)
            U_a = self.utility_a(X, Y)
            U_b = self.utility_b(self.total_x - X, self.total_y - Y)
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
        return self

    def add_endowment(
        self,
        x_endowment: float,
        y_endowment: float,
        *,
        label: str | Label = "e",
        color: str | None = None,
        marker: Marker | None = None,
    ) -> EdgeworthBox:
        """Mark the initial endowment point E.

        *label* is the text, or a :class:`Label` for its text, position,
        colour, and size (default ``theme.edgeworth_label``).
        """
        text, style = split_label(label, self.theme.edgeworth_label, "e")
        assert text is not None
        with styled(self, {}, markers={"endowment": marker}, labels={"endowment_label": style}):
            if not (0.0 <= x_endowment <= self.total_x and 0.0 <= y_endowment <= self.total_y):
                raise ValueError("Endowment must lie inside the Edgeworth box.")

            self.endowment = (float(x_endowment), float(y_endowment))
            endowment_marker = self.theme.endowment_marker
            c = color or endowment_marker.color or self.theme.eq_color
            size = endowment_marker.size if endowment_marker.size is not None else self.theme.eq_markersize
            plot_endowment(
                self.ax,
                x=x_endowment,
                y=y_endowment,
                total_x=self.total_x,
                total_y=self.total_y,
                color=c,
                markersize=size,
                label=text,
            )
        return self

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

            X, Y = self._grid(res=res)
            U_a = self.utility_a(X, Y)
            U_b = self.utility_b(self.total_x - X, self.total_y - Y)
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
        return self

    def _levels_around(self, anchor: float, n: int, spread: float) -> list[float]:
        return around_anchor_levels(anchor=float(anchor), n=n, spread=spread)

    def _focus_levels(
        self,
        *,
        anchor: float,
        u_min: float,
        u_max: float,
        n: int,
        spread: float,
        extra: float | None = None,
    ) -> list[float]:
        return focus_levels(
            anchor=anchor,
            u_min=u_min,
            u_max=u_max,
            n=n,
            spread=spread,
            extra=extra,
        )

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
            if px <= 0 or py <= 0:
                raise ValueError("px and py must be positive.")
            if self.walrasian_equilibrium is None:
                self.add_walrasian_equilibrium(px=px, py=py)

            equilibrium = self.walrasian_equilibrium
            assert equilibrium is not None
            x_star, y_star = equilibrium
            ua_star = self._eval_ua(x_star, y_star)
            ub_star = self._eval_ub(x_star, y_star)
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
            if px <= 0 or py <= 0:
                raise ValueError("px and py must be positive.")
            if self.walrasian_equilibrium is None:
                self.add_walrasian_equilibrium(px=px, py=py)

            equilibrium = self.walrasian_equilibrium
            assert equilibrium is not None
            x_star, y_star = equilibrium
            u_a_star = self._eval_ua(x_star, y_star)
            u_b_star = self._eval_ub(x_star, y_star)
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

    def _contract_curve_mrs(self, *, n: int, tolerance: float) -> np.ndarray:
        return contract_curve_mrs(
            utility_a=self.utility_a,
            utility_b=self.utility_b,
            total_x=self.total_x,
            total_y=self.total_y,
            n=n,
            tolerance=tolerance,
            eps=_EPS,
        )

    def _contract_curve_pareto(self, *, n: int) -> np.ndarray:
        return contract_curve_pareto(
            eval_ua=self._eval_ua,
            eval_ub=self._eval_ub,
            total_x=self.total_x,
            total_y=self.total_y,
            n=n,
            eps=_EPS,
        )

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
        return self

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
        apply_equilibrium_focus(self, px, py, config or EquilibriumFocusConfig())
        return self

    def _should_include_endowment_ic(self, *, min_relative_gap: float) -> bool:
        return should_include_endowment_ic(self, min_relative_gap=min_relative_gap)

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
            core = core_allocations(self, tol)

            self.core_points = self._unique_points(core)
            plot_core(
                self.ax,
                core_points=self.core_points,
                color=color or t.core_color,
                linewidth=linewidth if linewidth is not None else t.core_linewidth,
                label="Core",
                min_points=min_points,
            )
        return self

    def _line_box_intersections(self, px: float, py: float, income: float) -> list[tuple[float, float]]:
        return line_box_intersections(
            px=px,
            py=py,
            income=income,
            total_x=self.total_x,
            total_y=self.total_y,
        )

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
        return self

    def add_walrasian_equilibrium(
        self,
        px: float,
        py: float,
        *,
        color: str | None = None,
        marker: str | Marker | None = None,
        markersize: float | None = None,
        label: str | Label = r"X^*",
        contract_stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Approximate Walrasian equilibrium on the budget line and contract curve.

        Parameters
        ----------
        color : str, optional
            Marker colour. *None* → ``theme.walrasian_color``.
        markersize : float, optional
            Marker size. *None* → ``theme.walrasian_markersize``.
        contract_stroke : Stroke, optional
            Line style for the contract curve (default ``theme.contract_stroke``).
        marker : str or Marker, optional
            Marker shape (legacy), or a :class:`Marker` for colour, size, and
            shape (default ``theme.walrasian_marker``).
        label : str or Label
            Label text, or a :class:`Label` for its text, position, colour,
            and size (default ``theme.edgeworth_label``).
        """
        t = self.theme
        text, label_style = split_label(label, self.theme.edgeworth_label, r"X^*")
        assert text is not None
        marker_style = marker if isinstance(marker, Marker) else None
        default_shape = t.walrasian_marker.shape or "*"
        if marker_style is not None:
            shape = marker_style.shape or default_shape
        elif isinstance(marker, str):
            shape = marker
        else:
            shape = default_shape
        with styled(
            self,
            {"contract": contract_stroke},
            markers={"walrasian": marker_style},
            labels={"walrasian_label": label_style},
        ):
            x_star, y_star = locate_walrasian(self, px, py)

            self.walrasian_equilibrium = (x_star, y_star)
            plot_equilibrium_marker(
                self.ax,
                x=x_star,
                y=y_star,
                total_x=self.total_x,
                total_y=self.total_y,
                color=color or t.walrasian_color,
                marker=shape,
                markersize=markersize if markersize is not None else t.walrasian_markersize,
                label=text,
            )
        return self

    def check_point(
        self,
        x: float,
        y: float,
        *,
        px: float | None = None,
        py: float | None = None,
        tol: float = 1e-3,
    ) -> dict[str, bool]:
        """Return key checklist conditions at a candidate allocation."""
        return check_allocation(self, x, y, px, py, tol)

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

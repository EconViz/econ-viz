"""Edgeworth box diagram for two-consumer exchange economies."""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from ..contours import around_anchor_levels, percentile_levels
from ..io import save_figure
from ..themes import default as _default_theme
from ..themes.theme import Theme
from .edgeworth_compute import (
    contract_curve_mrs,
    contract_curve_pareto,
    focus_levels,
    line_box_intersections,
    mrs,
    unique_points,
    walrasian_equilibrium_point,
)
from .edgeworth_plotter import (
    plot_contract_curve,
    plot_core,
    plot_endowment,
    plot_equilibrium_marker,
    plot_indifference_pair,
    plot_price_line,
)
from .edgeworth_state import EdgeworthState

_MAX_DPI = 1200
_DEFAULT_DPI = 300
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
        title: str | None = None,
        dpi: int = _DEFAULT_DPI,
        theme: Theme = _default_theme,
        utility_a_color: str | None = None,
        utility_b_color: str | None = None,
    ):
        if total_x <= 0 or total_y <= 0:
            raise ValueError("total_x and total_y must be positive.")

        self.utility_a = utility_a
        self.utility_b = utility_b
        self.total_x = float(total_x)
        self.total_y = float(total_y)
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.dpi = max(1, min(int(dpi), _MAX_DPI))
        self.theme = theme
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

    def set_utility_colors(self, *, color_a: str, color_b: str) -> "EdgeworthBox":
        """Update default colors for utility A/B curves."""
        self.utility_a_color = color_a
        self.utility_b_color = color_b
        return self

    def _apply_base_style(self) -> None:
        t = self.theme
        self.ax.set_xlim(0.0, self.total_x)
        self.ax.set_ylim(0.0, self.total_y)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for side in ("top", "right", "bottom", "left"):
            self.ax.spines[side].set_visible(True)
            self.ax.spines[side].set_color(t.axis_color)
            self.ax.spines[side].set_linewidth(1.2)

        self.ax.set_xlabel(rf"${self.x_label}_A$", color=t.label_color)
        self.ax.set_ylabel(rf"${self.y_label}_A$", color=t.label_color)
        self.ax.text(0.0, 0.0, r"$O_A$", ha="right", va="top", color=t.label_color)
        self.ax.text(self.total_x, self.total_y, r"$O_B$", ha="left", va="bottom", color=t.label_color)
        self.ax.text(
            self.total_x * 0.98,
            self.total_y * -0.06,
            rf"${self.x_label}_B$",
            ha="right",
            va="top",
            color=t.label_color,
        )
        self.ax.text(
            self.total_x * -0.04,
            self.total_y * 0.98,
            rf"${self.y_label}_B$",
            ha="right",
            va="top",
            color=t.label_color,
            rotation=90,
        )
        if self.title:
            self.ax.set_title(self.title, color=t.label_color)

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
    ) -> "EdgeworthBox":
        """Draw both consumers' indifference maps."""
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
        label: str = "e",
        color: str | None = None,
    ) -> "EdgeworthBox":
        """Mark the initial endowment point E."""
        if not (0.0 <= x_endowment <= self.total_x and 0.0 <= y_endowment <= self.total_y):
            raise ValueError("Endowment must lie inside the Edgeworth box.")

        self.endowment = (float(x_endowment), float(y_endowment))
        c = color or self.theme.eq_color
        plot_endowment(
            self.ax,
            x=x_endowment,
            y=y_endowment,
            total_x=self.total_x,
            total_y=self.total_y,
            color=c,
            markersize=max(self.theme.eq_markersize, 6),
            label=label,
        )
        return self

    def add_endowment_indifference(
        self,
        *,
        color_a: str | None = None,
        color_b: str | None = None,
        linewidth: float | None = None,
        res: int = 300,
    ) -> "EdgeworthBox":
        """Draw each agent's indifference curve through the endowment point."""
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
    ) -> "EdgeworthBox":
        """Draw indifference curves around the Walrasian equilibrium utility levels."""
        if px <= 0 or py <= 0:
            raise ValueError("px and py must be positive.")
        if self.walrasian_equilibrium is None:
            self.add_walrasian_equilibrium(px=px, py=py)

        x_star, y_star = self.walrasian_equilibrium
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
    ) -> "EdgeworthBox":
        """Draw one indifference curve per agent through the Walrasian equilibrium."""
        if px <= 0 or py <= 0:
            raise ValueError("px and py must be positive.")
        if self.walrasian_equilibrium is None:
            self.add_walrasian_equilibrium(px=px, py=py)

        x_star, y_star = self.walrasian_equilibrium
        u_a_star = self._eval_ua(x_star, y_star)
        u_b_star = self._eval_ub(x_star, y_star)
        return self.add_indifference_curves(
            levels_a=[u_a_star],
            levels_b=[u_b_star],
            color_a=color_a,
            color_b=color_b,
            linewidth=linewidth,
            res=res,
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
        linewidth: float | None = 1.2,
        linestyle: str = "--",
        tolerance: float = 0.05,
        method: str = "auto",
    ) -> "EdgeworthBox":
        """Approximate and draw the contract curve."""
        if method not in {"auto", "mrs", "pareto"}:
            raise ValueError("method must be one of: auto, mrs, pareto.")

        points = np.empty((0, 2), dtype=float)
        if method in {"auto", "mrs"}:
            points = self._contract_curve_mrs(n=n, tolerance=tolerance)
        if len(points) < 4 and method in {"auto", "pareto"}:
            points = self._contract_curve_pareto(n=n)

        self.contract_curve_points = points
        c = color or self.theme.budget_color
        lw = linewidth if linewidth is not None else 1.2
        plot_contract_curve(
            self.ax,
            points=points,
            color=c,
            linewidth=lw,
            linestyle=linestyle,
            label="Contract curve",
        )
        return self

    def apply_equilibrium_focus(
        self,
        *,
        px: float,
        py: float,
        config: EquilibriumFocusConfig | None = None,
    ) -> "EdgeworthBox":
        """Render only the most informative indifference curves around equilibrium.

        Draws a bounded number of ICs per agent around ``X*`` (default: 3-5).
        Endowment ICs are optional and, when included, compete for the same cap.
        """
        cfg = config or EquilibriumFocusConfig()
        if self.walrasian_equilibrium is None:
            self.add_walrasian_equilibrium(px=px, py=py)

        min_curves = max(1, int(cfg.min_curves_per_agent))
        max_curves = max(min_curves, int(cfg.max_curves_per_agent))
        min_curves = max(3, min_curves)
        max_curves = min(5, max_curves)
        if min_curves > max_curves:
            min_curves = max_curves

        include = cfg.include_endowment_indifference
        if include == "auto":
            include = self._should_include_endowment_ic(min_relative_gap=cfg.min_relative_gap)

        if self.walrasian_equilibrium is None:
            raise ValueError("Walrasian equilibrium is required for equilibrium-focused rendering.")
        x_star, y_star = self.walrasian_equilibrium
        ua_star = self._eval_ua(x_star, y_star)
        ub_star = self._eval_ub(x_star, y_star)

        ua_e: float | None = None
        ub_e: float | None = None
        if bool(include) and self.endowment is not None:
            ex, ey = self.endowment
            ua_e = self._eval_ua(ex, ey)
            ub_e = self._eval_ub(ex, ey)

        target_n = min_curves + (1 if bool(include) and min_curves < max_curves else 0)

        X, Y = self._grid(res=cfg.res)
        U_a = self.utility_a(X, Y)
        U_b = self.utility_b(self.total_x - X, self.total_y - Y)
        ua_levels = self._focus_levels(
            anchor=ua_star,
            u_min=float(np.nanmin(U_a)),
            u_max=float(np.nanmax(U_a)),
            n=target_n,
            spread=cfg.equilibrium_spread,
            extra=ua_e if bool(include) else None,
        )
        ub_levels = self._focus_levels(
            anchor=ub_star,
            u_min=float(np.nanmin(U_b)),
            u_max=float(np.nanmax(U_b)),
            n=target_n,
            spread=cfg.equilibrium_spread,
            extra=ub_e if bool(include) else None,
        )

        lw_eq = cfg.equilibrium_linewidth
        if lw_eq is None:
            lw_eq = cfg.endowment_linewidth
        levels_a = ua_levels[:max_curves]
        levels_b = ub_levels[:max_curves]
        self.equilibrium_focus_levels_a = levels_a
        self.equilibrium_focus_levels_b = levels_b
        self.add_indifference_curves(
            levels_a=levels_a,
            levels_b=levels_b,
            linewidth=lw_eq,
            res=cfg.res,
        )
        return self

    def _should_include_endowment_ic(self, *, min_relative_gap: float) -> bool:
        if self.endowment is None or self.walrasian_equilibrium is None:
            return False
        ex, ey = self.endowment
        x_star, y_star = self.walrasian_equilibrium
        ua_e = self._eval_ua(ex, ey)
        ub_e = self._eval_ub(ex, ey)
        ua_s = self._eval_ua(x_star, y_star)
        ub_s = self._eval_ub(x_star, y_star)
        gap_a = abs(ua_s - ua_e) / (abs(ua_s) + 1e-9)
        gap_b = abs(ub_s - ub_e) / (abs(ub_s) + 1e-9)
        return max(gap_a, gap_b) >= max(min_relative_gap, 0.0)

    def add_core(
        self,
        *,
        color: str = "#C0392B",
        linewidth: float = 3.0,
        min_points: int = 2,
        tol: float = 1e-6,
    ) -> "EdgeworthBox":
        """Draw the core segment (IR part of the contract curve)."""
        if self.endowment is None:
            raise ValueError("Endowment is required. Call add_endowment(...) first.")
        if len(self.contract_curve_points) == 0:
            raise ValueError("Contract curve is required. Call add_contract_curve(...) first.")

        ex, ey = self.endowment
        ua_e = self._eval_ua(ex, ey)
        ub_e = self._eval_ub(ex, ey)

        core: list[tuple[float, float]] = []
        for x, y in self.contract_curve_points:
            if self._eval_ua(float(x), float(y)) >= ua_e - tol and self._eval_ub(float(x), float(y)) >= ub_e - tol:
                core.append((float(x), float(y)))

        self.core_points = self._unique_points(core)
        plot_core(
            self.ax,
            core_points=self.core_points,
            color=color,
            linewidth=linewidth,
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
        color: str = "#1F4BFF",
        linewidth: float = 2.0,
        label: str = "Price line",
    ) -> "EdgeworthBox":
        """Draw the price line through endowment with slope -px/py."""
        if px <= 0 or py <= 0:
            raise ValueError("px and py must be positive.")
        if self.endowment is None:
            raise ValueError("Endowment is required. Call add_endowment(...) first.")

        ex, ey = self.endowment
        income = px * ex + py * ey
        pts = self._line_box_intersections(px, py, income)
        plot_price_line(self.ax, points=pts, color=color, linewidth=linewidth, label=label)
        return self

    def add_walrasian_equilibrium(
        self,
        px: float,
        py: float,
        *,
        color: str = "#2E86AB",
        marker: str = "*",
        markersize: float = 10.0,
        label: str = r"X^*",
    ) -> "EdgeworthBox":
        """Approximate Walrasian equilibrium on the budget line and contract curve."""
        if px <= 0 or py <= 0:
            raise ValueError("px and py must be positive.")
        if self.endowment is None:
            raise ValueError("Endowment is required. Call add_endowment(...) first.")
        if len(self.contract_curve_points) == 0:
            self.add_contract_curve()

        ex, ey = self.endowment
        income = px * ex + py * ey

        candidates = self.contract_curve_points
        x_star, y_star = walrasian_equilibrium_point(
            candidates=candidates,
            px=px,
            py=py,
            income=income,
            mrs_a_fn=lambda x, y: self._mrs(self.utility_a, x, y),
            mrs_b_fn=lambda x, y: self._mrs(self.utility_b, self.total_x - x, self.total_y - y),
        )

        self.walrasian_equilibrium = (x_star, y_star)
        plot_equilibrium_marker(
            self.ax,
            x=x_star,
            y=y_star,
            total_x=self.total_x,
            total_y=self.total_y,
            color=color,
            marker=marker,
            markersize=markersize,
            label=label,
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
        checks: dict[str, bool] = {}
        checks["market_clearing"] = abs((x + (self.total_x - x)) - self.total_x) <= tol

        if self.endowment is not None:
            ex, ey = self.endowment
            checks["individual_rationality"] = (
                self._eval_ua(x, y) >= self._eval_ua(ex, ey) - tol
                and self._eval_ub(x, y) >= self._eval_ub(ex, ey) - tol
            )
        else:
            checks["individual_rationality"] = False

        if px is not None and py is not None and self.endowment is not None:
            ex, ey = self.endowment
            income = px * ex + py * ey
            checks["budget_balance"] = abs(px * x + py * y - income) <= tol * max(income, 1.0)
        else:
            checks["budget_balance"] = False

        mrs_a = self._mrs(self.utility_a, x, y)
        mrs_b = self._mrs(self.utility_b, self.total_x - x, self.total_y - y)
        checks["mrs_equal"] = (
            np.isfinite(mrs_a)
            and np.isfinite(mrs_b)
            and mrs_a > 0
            and mrs_b > 0
            and abs(np.log(mrs_a / mrs_b)) <= 0.08
        )
        return checks

    def show_legend(self, **kwargs) -> "EdgeworthBox":
        kwargs.setdefault("frameon", False)
        kwargs.setdefault("fontsize", 10)
        self.ax.legend(**kwargs)
        return self

    def save(self, path: str, **kwargs) -> None:
        """Export the Edgeworth box figure to disk."""
        save_figure(self.fig, path=path, dpi=self.dpi, close=True, **kwargs)

    def show(self) -> None:
        """Display the figure in an interactive matplotlib window."""
        self.fig.show()

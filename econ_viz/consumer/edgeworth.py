"""Edgeworth box diagram for two-consumer exchange economies."""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.levels import percentile
from ..exceptions import ExportError
from ..themes import default as _default_theme
from ..themes.theme import Theme

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

        self.endowment: tuple[float, float] | None = None
        self.contract_curve_points: np.ndarray = np.empty((0, 2), dtype=float)
        self.core_points: np.ndarray = np.empty((0, 2), dtype=float)
        self.walrasian_equilibrium: tuple[float, float] | None = None
        self.equilibrium_focus_levels_a: list[float] = []
        self.equilibrium_focus_levels_b: list[float] = []

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self._apply_base_style()

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
        x0 = min(max(float(x), _EPS), self.total_x - _EPS)
        y0 = min(max(float(y), _EPS), self.total_y - _EPS)
        ux = (float(func(x0 + h, y0)) - float(func(x0 - h, y0))) / (2.0 * h)
        uy = (float(func(x0, y0 + h)) - float(func(x0, y0 - h))) / (2.0 * h)
        if not np.isfinite(ux) or not np.isfinite(uy) or abs(uy) < 1e-9:
            return np.nan
        return ux / uy

    def _unique_points(self, points: list[tuple[float, float]], digits: int = 4) -> np.ndarray:
        if not points:
            return np.empty((0, 2), dtype=float)
        rounded = {(round(float(x), digits), round(float(y), digits)) for x, y in points}
        arr = np.array(sorted(rounded, key=lambda p: p[0]), dtype=float)
        return arr

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
        lv_a = percentile(U_a, n=levels_a) if isinstance(levels_a, int) else list(levels_a)
        lv_b = percentile(U_b, n=levels_b) if isinstance(levels_b, int) else list(levels_b)

        self.ax.contour(X, Y, U_a, levels=lv_a, colors=ca, linewidths=lw)
        self.ax.contour(X, Y, U_b, levels=lv_b, colors=cb, linewidths=lw, linestyles="--")
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
        self.ax.plot(
            x_endowment,
            y_endowment,
            "o",
            color=c,
            markersize=max(self.theme.eq_markersize, 6),
            zorder=20,
        )
        self.ax.text(
            x_endowment + self.total_x * 0.015,
            y_endowment + self.total_y * 0.015,
            rf"${label}$",
            color=c,
            fontsize=11,
            zorder=21,
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
        self.ax.contour(X, Y, U_a, levels=[u_a_e], colors=ca, linewidths=lw)
        self.ax.contour(X, Y, U_b, levels=[u_b_e], colors=cb, linewidths=lw, linestyles="--")
        return self

    def _levels_around(self, anchor: float, n: int, spread: float) -> list[float]:
        if n <= 1:
            return [float(anchor)]
        width = max(abs(float(anchor)), 1.0) * max(spread, 1e-6)
        lo = float(anchor) - width
        hi = float(anchor) + width
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        return np.linspace(lo, hi, n).tolist()

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
        anchor_f = float(anchor)
        u_min_f = float(u_min)
        u_max_f = float(u_max)
        if not np.isfinite(anchor_f) or not np.isfinite(u_min_f) or not np.isfinite(u_max_f):
            return [float(anchor)]
        if u_max_f <= u_min_f:
            return [float(anchor_f)]

        pad = max(abs(anchor_f), 1.0) * 1e-6
        lo_bound = u_min_f + pad
        hi_bound = u_max_f - pad
        if hi_bound <= lo_bound:
            return [float(np.clip(anchor_f, u_min_f, u_max_f))]

        width = max(abs(anchor_f), 1.0) * max(spread, 1e-6)
        lo = max(lo_bound, anchor_f - width)
        hi = min(hi_bound, anchor_f + width)
        if hi - lo <= 1e-12:
            full = hi_bound - lo_bound
            if full <= 1e-12:
                return [float(np.clip(anchor_f, u_min_f, u_max_f))]
            interval = min(max(2.0 * width, full * 0.2), full)
            lo = min(max(anchor_f - 0.5 * interval, lo_bound), hi_bound - interval)
            hi = lo + interval

        levels = np.linspace(lo, hi, n).tolist()
        if extra is not None and np.isfinite(extra):
            levels.append(float(np.clip(extra, lo_bound, hi_bound)))

        unique_levels: list[float] = []
        for lv in sorted(float(v) for v in levels if np.isfinite(v)):
            if not unique_levels or not np.isclose(unique_levels[-1], lv, rtol=1e-6, atol=1e-9):
                unique_levels.append(lv)
        return unique_levels

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
        x_grid = np.linspace(_EPS, self.total_x - _EPS, n)
        y_grid = np.linspace(_EPS, self.total_y - _EPS, max(3 * n, 180))
        points: list[tuple[float, float]] = []

        for x in x_grid:
            best_y: float | None = None
            best_score = np.inf
            for y in y_grid:
                mrs_a = self._mrs(self.utility_a, x, y)
                mrs_b = self._mrs(self.utility_b, self.total_x - x, self.total_y - y)
                if not np.isfinite(mrs_a) or not np.isfinite(mrs_b) or mrs_a <= 0 or mrs_b <= 0:
                    continue
                score = abs(np.log(mrs_a) - np.log(mrs_b))
                if score < best_score:
                    best_score = score
                    best_y = float(y)
            if best_y is not None and best_score <= tolerance:
                points.append((float(x), best_y))
        return self._unique_points(points)

    def _contract_curve_pareto(self, *, n: int) -> np.ndarray:
        try:
            from scipy.optimize import minimize
        except Exception:
            return np.empty((0, 2), dtype=float)

        lambdas = np.linspace(0.01, 0.99, n)
        points: list[tuple[float, float]] = []
        prev = np.array([self.total_x * 0.5, self.total_y * 0.5], dtype=float)

        bounds = [(_EPS, self.total_x - _EPS), (_EPS, self.total_y - _EPS)]

        for lam in lambdas:
            starts = [
                prev,
                np.array([self.total_x * 0.25, self.total_y * 0.25]),
                np.array([self.total_x * 0.75, self.total_y * 0.75]),
            ]
            best = None
            best_obj = np.inf

            def obj(v: np.ndarray) -> float:
                x, y = float(v[0]), float(v[1])
                ua = self._eval_ua(x, y)
                ub = self._eval_ub(x, y)
                if not np.isfinite(ua) or not np.isfinite(ub):
                    return 1e9
                return -(lam * ua + (1.0 - lam) * ub)

            for x0 in starts:
                res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds)
                if res.success and float(res.fun) < best_obj:
                    best_obj = float(res.fun)
                    best = res.x

            if best is not None:
                prev = np.asarray(best, dtype=float)
                points.append((float(best[0]), float(best[1])))

        return self._unique_points(points)

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
        if len(points) > 0:
            self.ax.plot(
                points[:, 0],
                points[:, 1],
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
        if len(self.core_points) >= min_points:
            self.ax.plot(self.core_points[:, 0], self.core_points[:, 1], color=color, linewidth=linewidth, label="Core")
        elif len(self.core_points) == 1:
            self.ax.plot(self.core_points[0, 0], self.core_points[0, 1], "o", color=color, label="Core")
        return self

    def _line_box_intersections(self, px: float, py: float, income: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []

        y0 = income / py
        if 0.0 <= y0 <= self.total_y:
            pts.append((0.0, y0))
        yw = (income - px * self.total_x) / py
        if 0.0 <= yw <= self.total_y:
            pts.append((self.total_x, yw))
        x0 = income / px
        if 0.0 <= x0 <= self.total_x:
            pts.append((x0, 0.0))
        xh = (income - py * self.total_y) / px
        if 0.0 <= xh <= self.total_x:
            pts.append((xh, self.total_y))

        unique = self._unique_points(pts, digits=8)
        return [(float(x), float(y)) for x, y in unique]

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
        if len(pts) >= 2:
            self.ax.plot(
                [pts[0][0], pts[-1][0]],
                [pts[0][1], pts[-1][1]],
                color=color,
                linewidth=linewidth,
                label=label,
            )
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
        price_ratio = px / py

        candidates = self.contract_curve_points
        if len(candidates) == 0:
            raise ValueError("Unable to compute contract curve for equilibrium search.")

        def score(p: np.ndarray) -> float:
            x, y = float(p[0]), float(p[1])
            budget_err = abs(px * x + py * y - income) / max(income, 1.0)
            mrs_a = self._mrs(self.utility_a, x, y)
            mrs_b = self._mrs(self.utility_b, self.total_x - x, self.total_y - y)
            mrs_err = 0.0
            if np.isfinite(mrs_a) and np.isfinite(mrs_b) and mrs_a > 0 and mrs_b > 0:
                mrs_err = abs(np.log(mrs_a / price_ratio)) + abs(np.log(mrs_b / price_ratio))
            return budget_err + 0.25 * mrs_err

        residuals = np.array([px * p[0] + py * p[1] - income for p in candidates], dtype=float)

        x_star: float
        y_star: float
        crossing_idx = np.where(residuals[:-1] * residuals[1:] <= 0.0)[0]
        if len(crossing_idx) > 0:
            i = int(crossing_idx[np.argmin(np.abs(residuals[crossing_idx]))])
            p0 = candidates[i]
            p1 = candidates[i + 1]
            r0 = float(residuals[i])
            r1 = float(residuals[i + 1])
            if abs(r1 - r0) > 1e-12:
                t = -r0 / (r1 - r0)
                t = min(max(t, 0.0), 1.0)
                p = p0 + t * (p1 - p0)
                x_star = float(p[0])
                y_star = float(p[1])
            else:
                x_star = float(p0[0])
                y_star = float(p0[1])
        else:
            idx = int(np.argmin([score(p) for p in candidates]))
            x_star = float(candidates[idx, 0])
            y_star = float(candidates[idx, 1])

        self.walrasian_equilibrium = (x_star, y_star)

        self.ax.plot(
            x_star,
            y_star,
            marker=marker,
            color=color,
            markersize=markersize,
            label=rf"${label}$",
            zorder=22,
        )
        self.ax.text(
            x_star + self.total_x * 0.012,
            y_star + self.total_y * 0.012,
            rf"${label}$",
            color=color,
            fontsize=11,
            zorder=23,
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
        try:
            self.fig.savefig(
                path,
                dpi=self.dpi,
                transparent=True,
                bbox_inches="tight",
                **kwargs,
            )
        except OSError as exc:
            raise ExportError(f"Failed to write '{path}': {exc}") from exc
        finally:
            plt.close(self.fig)

    def show(self) -> None:
        """Display the figure in an interactive matplotlib window."""
        self.fig.show()

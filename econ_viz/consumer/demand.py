"""Linked multi-panel Marshallian demand teaching diagram."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import matplotlib.lines as mlines
import numpy as np

from ..canvas.figure import Figure
from ..canvas.legend import place_legend
from ..canvas.stroke import apply_strokes, styled, tag
from ..enums import Layout, UtilityType
from ..models import PerfectSubstitutes
from ..optimizer import solve
from ..optimizer.solver import Equilibrium
from ..themes.axis import Axis
from ..themes.label import Label, split_label
from ..themes.legend import Legend
from ..themes.marker import Marker
from ..themes.stroke import Stroke
from .paths import LinearBudget, PricePath, QuantityAxis

_GOODS_SPACE_PADDING = 1.18
_PRICE_SYMBOL = {
    "px": r"p_{x}",
    "py": r"p_{y}",
}


class _UtilityDrawOptions(TypedDict, total=False):
    show_kinks: bool
    kink_radius: float
    res: int


class DemandDiagram(Figure):
    """Two-panel diagram linking utility maximisation to Marshallian demand.

    *x_axis* / *y_axis* (:class:`Axis`) set both panels' axis strokes and the
    goods-space panel's labels; the demand panel keeps its quantity and price
    labels. Other keyword arguments go to :class:`Figure`.
    """

    def __init__(
        self,
        path: PricePath,
        x_max: float | None = None,
        y_max: float | None = None,
        demand_x_max: float | None = None,
        demand_y_max: float | None = None,
        x_label: str = "x",
        y_label: str = "y",
        title: str | None = None,
        x_axis: Axis | None = None,
        y_axis: Axis | None = None,
        **kwargs,
    ):
        if path.parameter_name not in {"px", "py"}:
            raise ValueError("DemandDiagram requires a price sweep path.")
        self.path = path
        self.func = path.func
        top_x_max, top_y_max = self._goods_space_bounds(x_max=x_max, y_max=y_max)
        super().__init__(
            Layout.STACKED,
            x_max=top_x_max,
            y_max=top_y_max,
            x_label=x_label,
            y_label=y_label,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            **kwargs,
        )
        self.utility_canvas = self[0]
        self.demand_canvas = self[1]
        quantity_axis = self._quantity_axis()
        self.demand_canvas.x_max = demand_x_max or max(self.path.quantity_values(quantity_axis)) * 1.15
        self.demand_canvas.y_max = demand_y_max or max(self.path.parameter_values) * 1.15
        self.demand_canvas.x_label = quantity_axis
        self.demand_canvas.y_label = _PRICE_SYMBOL[self.path.parameter_name]
        self.demand_canvas.ax.cla()
        self.demand_canvas._apply_base_style()

    def add_marshallian_panel(
        self,
        price_markers: list[float] | None = None,
        show_pcc: bool = False,
        show_demand_guides: bool = True,
        show_legend: bool = True,
        curve_stroke: Stroke | None = None,
        budget_stroke: Stroke | None = None,
        drop_stroke: Stroke | None = None,
        path_stroke: Stroke | None = None,
        demand_stroke: Stroke | None = None,
        guide_stroke: Stroke | None = None,
        point_marker: Marker | None = None,
        point_label: Label | None = None,
        legend: Legend | None = None,
    ) -> DemandDiagram:
        """Draw the goods-space panel and the linked Marshallian demand panel.

        The ``*_stroke`` arguments restyle one kind of line each: indifference
        curves, budget lines, equilibrium drop lines, and the PCC in the top
        panel; the demand curve and its guides in the bottom panel.
        ``point_marker`` restyles the equilibrium points in both panels, and
        ``point_label`` their labels (its *text* is ignored; default
        ``theme.point_label``). ``legend`` places both panels' legends (default
        ``theme.legend``: where they cover the least of each panel).
        """
        price_markers = price_markers or [self.path.parameter_values[len(self.path.parameter_values) // 2]]
        quantity_axis = self._quantity_axis()
        selected_equilibria = [self._solve_at_parameter(price) for price in price_markers]
        self._reset_goods_canvas(selected_equilibria)
        top = {"curve": curve_stroke, "budget": budget_stroke, "drop": drop_stroke, "path": path_stroke}
        bottom = {"demand": demand_stroke, "guide": guide_stroke}
        # Style before the legends are built: a legend copies line styles when created.
        with (
            styled(
                self.utility_canvas,
                top,
                markers={"equilibrium": point_marker},
                labels={"equilibrium_label": split_label(point_label, self.utility_canvas.theme.point_label)[1]},
            ),
            styled(
                self.demand_canvas,
                bottom,
                markers={"point": point_marker, "tie": point_marker},
                labels={"point_label": split_label(point_label, self.demand_canvas.theme.point_label)[1]},
            ),
        ):
            selected_levels = sorted(dict.fromkeys(eq.utility for _, eq in selected_equilibria))
            self.utility_canvas.add_utility(
                self.func,
                levels=selected_levels,
                label="IC",
                **self._utility_draw_options(),
            )

            for idx, (price, solved) in enumerate(zip(price_markers, selected_equilibria, strict=True)):
                budget, eq = solved
                line_style = "-" if idx == 0 else "--"
                point_text = self._marker_label(idx)
                price_symbol = _PRICE_SYMBOL[self.path.parameter_name]
                line_label = f"{point_text}: {price_symbol}={price:.2g}"
                self.utility_canvas.add_budget(
                    px=budget.px,
                    py=budget.py,
                    income=budget.income,
                    linestyle=line_style,
                    label=line_label,
                )
                self.utility_canvas.add_equilibrium(eq, label=point_text)
                quantity = eq.x if quantity_axis == "x" else eq.y
                self.demand_canvas.add_point(quantity, price, label=point_text)
                if show_demand_guides:
                    self._add_demand_guides(quantity=quantity, price=price)

            self._add_demand_curve(label="Marshallian demand")
            if show_pcc:
                self.utility_canvas.add_path(self.path, label="PCC", show_points=False)
        if show_legend:
            self.utility_canvas.show_legend(legend=legend)
            handles = [
                mlines.Line2D([], [], color=self.demand_canvas.theme.ic_color, label="Marshallian demand"),
                mlines.Line2D(
                    [], [], color=self.demand_canvas.theme.ic_color, linestyle="--", label="corner / boundary"
                ),
            ]
            if demand_stroke is not None:
                for handle in handles:
                    tag(handle, "demand")
                apply_strokes(self.demand_canvas.ax, handles, bottom)
            place_legend(
                self.demand_canvas.ax,
                handles,
                [str(h.get_label()) for h in handles],
                (legend or Legend()).merged_over(self.demand_canvas.theme.legend),
            )
        else:
            for ax in (self.utility_canvas.ax, self.demand_canvas.ax):
                existing = ax.get_legend()
                if existing is not None:
                    existing.remove()
        return self

    def _quantity_axis(self) -> QuantityAxis:
        return "x" if self.path.parameter_name == "px" else "y"

    def _marker_label(self, idx: int) -> str:
        return chr(ord("A") + idx)

    def _utility_draw_options(self) -> _UtilityDrawOptions:
        options: _UtilityDrawOptions = {}
        utility_type = getattr(self.func, "utility_type", None)
        if utility_type is UtilityType.KINKED:
            options["show_kinks"] = True
            options["kink_radius"] = 1.2
        if self.func.__class__.__name__ == "QuasiLinear":
            options["res"] = 900
        return options

    def _goods_space_bounds(
        self,
        *,
        x_max: float | None,
        y_max: float | None,
    ) -> tuple[float, float]:
        if x_max is not None and y_max is not None:
            return x_max, y_max

        x_intercepts = [budget.income / budget.px for budget in self.path.budgets]
        y_intercepts = [budget.income / budget.py for budget in self.path.budgets]
        eq_xs = [eq.x for eq in self.path.equilibria]
        eq_ys = [eq.y for eq in self.path.equilibria]

        max_x = max([*x_intercepts, *eq_xs, 1.0])
        max_y = max([*y_intercepts, *eq_ys, 1.0])
        resolved_x_max = x_max or max_x * _GOODS_SPACE_PADDING
        resolved_y_max = y_max or max_y * _GOODS_SPACE_PADDING
        return resolved_x_max, resolved_y_max

    def _reset_goods_canvas(self, solved_points: list[tuple[LinearBudget, Equilibrium]]) -> None:
        x_intercepts = [budget.income / budget.px for budget, _ in solved_points]
        y_intercepts = [budget.income / budget.py for budget, _ in solved_points]
        eq_xs = [eq.x for _, eq in solved_points]
        eq_ys = [eq.y for _, eq in solved_points]

        self.utility_canvas.x_max = max([*x_intercepts, *eq_xs, 1.0]) * _GOODS_SPACE_PADDING
        self.utility_canvas.y_max = max([*y_intercepts, *eq_ys, 1.0]) * _GOODS_SPACE_PADDING
        self.utility_canvas.ax.clear()
        self.utility_canvas._legend_handles.clear()
        self.utility_canvas._apply_base_style()

    def _solve_at_parameter(self, value: float) -> tuple[LinearBudget, Equilibrium]:
        budget = self.path.base_budget.with_update(**{self.path.parameter_name: value})
        eq = solve(self.func, px=budget.px, py=budget.py, income=budget.income)
        return budget, eq

    def _add_demand_curve(self, label: str) -> None:
        if getattr(self.func, "utility_type", None) is UtilityType.LINEAR:
            self._add_linear_demand_curve(label=label)
            return

        quantity_axis = self._quantity_axis()
        xs = np.array(self.path.quantity_values(quantity_axis), dtype=float)
        ys = np.array(self.path.parameter_values, dtype=float)
        bundle_types = [eq.bundle_type for eq in self.path.equilibria]
        connected = self._connected_mask(quantity_axis)

        segment_start = 0
        for idx in range(1, len(xs)):
            if connected[idx - 1]:
                continue
            self._plot_segment(xs, ys, bundle_types, segment_start, idx)
            segment_start = idx
        self._plot_segment(xs, ys, bundle_types, segment_start, len(xs))

        self.demand_canvas._legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=self.demand_canvas.theme.ic_color,
                linewidth=self.demand_canvas.theme.ic_linewidth,
                label=label,
            )
        )

    def _add_linear_demand_curve(self, label: str) -> None:
        quantity_axis = self._quantity_axis()
        xs = np.array(self.path.quantity_values(quantity_axis), dtype=float)
        ys = np.array(self.path.parameter_values, dtype=float)
        preferred = [self._preferred_corner(budget) for budget in self.path.budgets]
        func = cast(PerfectSubstitutes, self.func)

        if quantity_axis == "x":
            chosen = np.array([side == "x" for side in preferred], dtype=bool)
            tie_price = func.a * self.path.base_budget.py / func.b
            tie_quantity = self.path.base_budget.income / tie_price
        else:
            chosen = np.array([side == "y" for side in preferred], dtype=bool)
            tie_price = func.b * self.path.base_budget.px / func.a
            tie_quantity = self.path.base_budget.income / tie_price

        if np.any(chosen):
            branch_xs = xs[chosen]
            branch_ys = ys[chosen]
            if ys.min() <= tie_price <= ys.max():
                branch_xs = np.append(branch_xs, tie_quantity)
                branch_ys = np.append(branch_ys, tie_price)
            (line,) = self.demand_canvas.ax.plot(
                branch_xs,
                branch_ys,
                color=self.demand_canvas.theme.ic_color,
                linewidth=self.demand_canvas.theme.ic_linewidth,
            )
            tag(line, "demand")

        if ys.min() <= tie_price <= ys.max():
            (line,) = self.demand_canvas.ax.plot(
                [0.0, tie_quantity],
                [tie_price, tie_price],
                color=self.demand_canvas.theme.ic_color,
                linewidth=self.demand_canvas.theme.ic_linewidth,
            )
            tag(line, "demand")
            self._add_tie_marker(tie_quantity, tie_price)

        self.demand_canvas._legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=self.demand_canvas.theme.ic_color,
                linewidth=self.demand_canvas.theme.ic_linewidth,
                label=label,
            )
        )

    def _connected_mask(self, quantity_axis: QuantityAxis) -> list[bool]:
        if len(self.path.equilibria) < 2:
            return []

        if getattr(self.func, "utility_type", None) is UtilityType.LINEAR:
            preferred = [self._preferred_corner(budget) for budget in self.path.budgets]
            return [left == right and left != "tie" for left, right in zip(preferred, preferred[1:], strict=False)]

        quantities = np.array(self.path.quantity_values(quantity_axis), dtype=float)
        connected: list[bool] = []
        for left, right in zip(quantities, quantities[1:], strict=False):
            connected.append(np.isfinite(left) and np.isfinite(right))
        return connected

    def _preferred_corner(self, budget: LinearBudget) -> str:
        x_corner = budget.income / budget.px
        y_corner = budget.income / budget.py
        u_x = float(self.func(x_corner, 0))
        u_y = float(self.func(0, y_corner))
        if np.isclose(u_x, u_y, rtol=1e-9, atol=1e-9):
            return "tie"
        return "x" if u_x > u_y else "y"

    def _plot_segment(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        bundle_types: list[str],
        start: int,
        stop: int,
    ) -> None:
        if stop - start <= 0:
            return
        linestyle = "--" if bundle_types[start] in {"corner", "boundary"} else "-"
        (line,) = self.demand_canvas.ax.plot(
            xs[start:stop],
            ys[start:stop],
            color=self.demand_canvas.theme.ic_color,
            linewidth=self.demand_canvas.theme.ic_linewidth,
            linestyle=linestyle,
        )
        tag(line, "demand")

    def _add_demand_guides(self, *, quantity: float, price: float) -> None:
        guide_style: dict[str, Any] = dict(
            color=self.demand_canvas.theme.eq_color,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )
        for xs, ys in (([quantity, quantity], [0, price]), ([0, quantity], [price, price])):
            (line,) = self.demand_canvas.ax.plot(xs, ys, **guide_style)
            tag(line, "guide")

    def _add_tie_marker(self, quantity: float, price: float) -> None:
        (point,) = self.demand_canvas.ax.plot(
            quantity,
            price,
            marker="o",
            linestyle="None",
            markersize=max(self.demand_canvas.theme.eq_markersize + 2, 7),
            color=self.demand_canvas.theme.ic_color,
            clip_on=False,
            zorder=6,
        )
        tag(point, "tie")

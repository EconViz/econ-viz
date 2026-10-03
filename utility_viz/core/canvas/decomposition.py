"""Hicks/Slutsky price-effect decomposition layer."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import matplotlib.lines as mlines
import numpy as np

from utility_viz.core.canvas.layers import LayerMixin
from utility_viz.core.canvas.legend import LegendMixin
from utility_viz.core.canvas.renderers import render_decomposition
from utility_viz.core.rendering.effect import Effect
from utility_viz.core.rendering.stroke import styled, tag
from utility_viz.core.styles.label import Label, split_label
from utility_viz.core.styles.legend import Legend
from utility_viz.core.styles.marker import Marker
from utility_viz.core.styles.stroke import Stroke
from utility_viz.enums import LineStyle

if TYPE_CHECKING:
    from typing_extensions import Self


class DecompositionMixin(LayerMixin, LegendMixin):
    """``add_decomposition`` for :class:`~utility_viz.core.canvas.base.Canvas`."""

    def add_decomposition(
        self,
        decomposition,
        *,
        show_arrows: bool = True,
        label_effects: bool = True,
        show_x_projections: bool = False,
        point_color: str | None = None,
        point_markersize: float | None = None,
        original_budget_color: str | None = None,
        original_budget_linewidth: float | None = None,
        original_budget_linestyle: str | None = None,
        compensated_budget_color: str | None = None,
        compensated_budget_linewidth: float | None = None,
        compensated_budget_linestyle: str | None = None,
        final_budget_color: str | None = None,
        final_budget_linewidth: float | None = None,
        final_budget_linestyle: str | None = None,
        substitution_color: str | None = None,
        income_color: str | None = None,
        effect_arrow_linewidth: float | None = None,
        original_budget_stroke: Stroke | None = None,
        compensated_budget_stroke: Stroke | None = None,
        final_budget_stroke: Stroke | None = None,
        substitution_stroke: Stroke | None = None,
        income_stroke: Stroke | None = None,
        projection_stroke: Stroke | None = None,
        guide_stroke: Stroke | None = None,
        range_stroke: Stroke | None = None,
        substitution: Effect | None = None,
        income: Effect | None = None,
        point_marker: Marker | None = None,
        point_label: Label | None = None,
        show_curves: bool = True,
        curve_stroke: Stroke | None = None,
        curve_label: Label | None = None,
        legend: Legend | None = None,
    ) -> Self:
        """Render a Hicks/Slutsky price-effect decomposition on this canvas.

        Prefer the ``*_stroke`` arguments for line styling. The
        ``*_budget_color`` / ``*_linewidth`` / ``*_linestyle``,
        *substitution_color*, *income_color*, and *effect_arrow_linewidth*
        arguments are shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
        decomposition : PriceEffectDecomposition
            Result returned by :func:`utility_viz.models.optimization.decompose_price_effect`.
        show_arrows : bool
            If ``True`` draw substitution/income arrows.
        label_effects : bool
            If ``True`` annotate effect magnitudes near each arrow.
        show_x_projections : bool
            If ``True`` draw x-axis projection guides and brackets.
        original_budget_stroke : Stroke, optional
            Line style for the original budget line (default ``theme.budget_stroke``).
        compensated_budget_stroke : Stroke, optional
            Line style for the compensated budget line (default ``theme.compensated_budget_stroke``).
        final_budget_stroke : Stroke, optional
            Line style for the final budget line (default ``theme.final_budget_stroke``).
        substitution_stroke : Stroke, optional
            Line style for the substitution-effect arrow (default ``theme.substitution_stroke``).
        income_stroke : Stroke, optional
            Line style for the income-effect arrow (default ``theme.income_stroke``).
        projection_stroke : Stroke, optional
            Line style for vertical guides from A, B, C to the x-axis (default ``theme.projection_stroke``).
        guide_stroke : Stroke, optional
            Line style for guides below the x-axis (default ``theme.guide_stroke``).
        range_stroke : Stroke, optional
            Line style for effect-size arrows below the x-axis.
        substitution, income : Effect, optional
            Colour, range-arrow height, and label of each effect. The colour
            overrides *substitution_color* / *income_color*; a Stroke colour
            overrides both.
        point_marker : Marker, optional
            Colour, size, and shape of bundles A, B, C and their legend entries (default ``theme.eq_marker``).
        point_label : Label, optional
            Position, colour, and size of the A, B, C labels (default
            ``theme.bundle_label``); ``Label(visible=False)`` hides them. Its
            *text* is ignored.
        show_curves : bool
            Draw the indifference curves through A (:math:`U_0`) and C
            (:math:`U_1 = V(p_1, I)`), plus the one through B for Slutsky, where
            B is off :math:`U_0` (default ``True``). Pass ``False`` when you
            draw the curves yourself with :meth:`add_utility`.
        curve_stroke : Stroke, optional
            Line style for those curves (default ``theme.ic_stroke``).
        curve_label : Label, optional
            Turns on the :math:`U_0`, :math:`U_1`, :math:`U_B` labels at the
            curves' right ends and sets their position, colour, and size (default
            ``theme.ic_label``). Its *text* is ignored.
        legend : Legend, optional
            Where and how the A, B, C and effect legend is drawn (default
            ``theme.legend``: placed where it covers the least of the diagram).
            ``Legend(visible=False)`` hides it.
        """
        if show_curves and decomposition.func is not None:
            self._add_decomposition_curves(decomposition, curve_stroke, curve_label)
        _, bundle_style = split_label(point_label, self.theme.bundle_label)
        substitution_color = _effect_color(substitution, substitution_color)
        income_color = _effect_color(income, income_color)
        t = self.theme
        with styled(
            self,
            {
                "original_budget": original_budget_stroke,
                "compensated_budget": compensated_budget_stroke,
                "final_budget": final_budget_stroke,
                "substitution": substitution_stroke,
                "income": income_stroke,
                "projection": projection_stroke,
                "guide": guide_stroke,
                "range": range_stroke,
            },
            markers={"bundle": point_marker},
            labels={"bundle_label": bundle_style},
        ):
            render_decomposition(
                self.ax,
                decomposition=decomposition,
                **_budget_options(
                    t,
                    original=(original_budget_color, original_budget_linewidth, original_budget_linestyle),
                    compensated=(compensated_budget_color, compensated_budget_linewidth, compensated_budget_linestyle),
                    final=(final_budget_color, final_budget_linewidth, final_budget_linestyle),
                ),
                **_point_options(t, point_color, point_markersize),
                **_effect_options(t, substitution_color, income_color, effect_arrow_linewidth),
                show_arrows=show_arrows,
                arrows_below_axis=show_x_projections,
                show_x_projections=show_x_projections,
                substitution_effect=substitution,
                income_effect=income,
                effect_label=t.effect_label,
            )
            if label_effects:
                self._register_decomposition_handles(
                    _decomposition_legend_handles(
                        decomposition,
                        point_color=point_color or t.eq_color,
                        point_markersize=point_markersize if point_markersize is not None else t.eq_markersize,
                        substitution_color=substitution_color or t.sub_effect_color,
                        income_color=income_color or t.inc_effect_color,
                        effect_linewidth=effect_arrow_linewidth
                        if effect_arrow_linewidth is not None
                        else t.effect_arrow_linewidth,
                        substitution_opacity=substitution.opacity if substitution is not None else None,
                        income_opacity=income.opacity if income is not None else None,
                    )
                )
                self.show_legend(legend=legend)
        return self

    def _register_decomposition_handles(self, handles: list[mlines.Line2D]) -> None:
        """Queue the A, B, C, substitution and income proxies for the legend, tagged by role."""
        self._legend_handles.extend(handles)
        for handle in self._legend_handles[-5:-2]:
            tag(handle, "bundle")
        tag(self._legend_handles[-2], "substitution")
        tag(self._legend_handles[-1], "income")

    def _add_decomposition_curves(self, decomposition, stroke: Stroke | None, label: Label | None) -> None:
        """Draw U0 through A, U1 through C, and the curve through B when it is off U0."""
        drawn: list[float] = []
        for name, bundle in (("U_0", decomposition.A), ("U_1", decomposition.C), ("U_B", decomposition.B)):
            level = float(bundle.utility)
            # Under Hicks, B lies on U0; skip levels already drawn.
            if any(np.isclose(level, u, rtol=1e-9, atol=1e-12) for u in drawn):
                continue
            drawn.append(level)
            self.add_utility(
                decomposition.func,
                levels=[level],
                show_bliss=False,
                stroke=stroke,
                ic_label=replace(label, text=f"${name}$") if label is not None else None,
            )


def _effect_color(effect: Effect | None, fallback: str | None) -> str | None:
    """An :class:`Effect` colour overrides the shorthand colour argument."""
    return effect.color if effect is not None and effect.color is not None else fallback


def _budget_options(t, *, original: tuple, compensated: tuple, final: tuple) -> dict:
    """Colour, width and style of the three budget lines (``(color, linewidth, linestyle)`` overrides)."""
    o_color, o_width, o_style = original
    c_color, c_width, c_style = compensated
    f_color, f_width, f_style = final
    return {
        "original_budget_color": o_color or t.budget_color,
        "original_budget_linewidth": o_width if o_width is not None else t.budget_linewidth,
        # Theme.budget_stroke always normalises style to a LineStyle.
        "original_budget_linestyle": o_style or cast(LineStyle, t.budget_stroke.style).value,
        "compensated_budget_color": c_color or t.compensated_budget_color,
        "compensated_budget_linewidth": c_width if c_width is not None else t.compensated_budget_linewidth,
        "compensated_budget_linestyle": c_style if c_style is not None else t.compensated_budget_linestyle,
        "final_budget_color": f_color or t.budget_color,
        "final_budget_linewidth": f_width if f_width is not None else t.budget_linewidth,
        "final_budget_linestyle": f_style or cast(LineStyle, t.final_budget_stroke.style).value,
    }


def _point_options(t, color: str | None, markersize: float | None) -> dict:
    """Colour, size and shape of bundles A, B and C."""
    return {
        "point_color": color or t.eq_color,
        "point_markersize": markersize if markersize is not None else t.eq_markersize,
        "point_marker_shape": t.eq_marker.shape or "o",
    }


def _effect_options(t, substitution_color: str | None, income_color: str | None, linewidth: float | None) -> dict:
    """Colour, width and style of the substitution and income arrows."""
    return {
        "substitution_color": substitution_color or t.sub_effect_color,
        "income_color": income_color or t.inc_effect_color,
        "effect_arrow_linewidth": linewidth if linewidth is not None else t.effect_arrow_linewidth,
        # Theme.substitution_stroke / income_stroke always normalise style to a LineStyle.
        "substitution_linestyle": cast(LineStyle, t.substitution_stroke.style).value,
        "income_linestyle": cast(LineStyle, t.income_stroke.style).value,
    }


def _bundle_handle(decomposition, name: str, color: str, markersize: float) -> mlines.Line2D:
    bundle = getattr(decomposition, name)
    return mlines.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="None",
        markersize=markersize,
        label=rf"${name}=({bundle.x:.2f},{bundle.y:.2f})$",
    )


def _effect_handle(color: str, linewidth: float, label: str, opacity: float | None) -> mlines.Line2D:
    return mlines.Line2D([], [], color=color, linestyle="--", linewidth=linewidth, label=label, alpha=opacity)


def _decomposition_legend_handles(
    decomposition,
    *,
    point_color: str,
    point_markersize: float,
    substitution_color: str,
    income_color: str,
    effect_linewidth: float,
    substitution_opacity: float | None,
    income_opacity: float | None,
) -> list[mlines.Line2D]:
    """Legend proxies: bundles A, B, C then the substitution and income effects."""
    sub_dx, sub_dy = decomposition.substitution_effect
    inc_dx, inc_dy = decomposition.income_effect
    return [
        *(_bundle_handle(decomposition, name, point_color, point_markersize) for name in ("A", "B", "C")),
        _effect_handle(
            substitution_color,
            effect_linewidth,
            rf"$Sub:\ \Delta x={sub_dx:+.2f},\ \Delta y={sub_dy:+.2f}$",
            substitution_opacity,
        ),
        _effect_handle(
            income_color,
            effect_linewidth,
            rf"$Inc:\ \Delta x={inc_dx:+.2f},\ \Delta y={inc_dy:+.2f}$",
            income_opacity,
        ),
    ]

"""Renderer for price-effect decomposition overlays."""

from __future__ import annotations

from typing import Iterable

from ...constants.canvas import INCOME_RANGE_Y, SUBSTITUTION_RANGE_Y
from ...enums import LabelPosition
from ...themes.label import Label
from ..effect import Effect
from ..labels import placement
from ..primitives import annotate_math, plot_point
from .budget import render_budget


def render_decomposition(
    ax,
    *,
    decomposition,
    point_color: str,
    point_markersize: float,
    original_budget_color: str,
    original_budget_linewidth: float,
    original_budget_linestyle: str,
    compensated_budget_color: str,
    compensated_budget_linewidth: float,
    compensated_budget_linestyle: str,
    final_budget_color: str,
    final_budget_linewidth: float,
    final_budget_linestyle: str,
    show_arrows: bool,
    arrows_below_axis: bool,
    substitution_color: str,
    income_color: str,
    effect_arrow_linewidth: float,
    show_x_projections: bool,
    substitution_effect: Effect | None = None,
    income_effect: Effect | None = None,
    effect_label: Label | None = None,
) -> None:
    """Render A/B/C bundles, budget lines, and effect arrows."""
    render_budget(
        ax,
        px=decomposition.px_before,
        py=decomposition.py,
        income=decomposition.income,
        color=original_budget_color,
        linewidth=original_budget_linewidth,
        linestyle=original_budget_linestyle,
        label=None,
        fill=False,
        fill_alpha=0.0,
    )
    _retag_last_budget(ax, "original_budget")
    render_budget(
        ax,
        px=decomposition.px_after,
        py=decomposition.py,
        income=decomposition.compensated_income,
        color=compensated_budget_color,
        linewidth=compensated_budget_linewidth,
        linestyle=compensated_budget_linestyle,
        label=None,
        fill=False,
        fill_alpha=0.0,
    )
    _retag_last_budget(ax, "compensated_budget")
    render_budget(
        ax,
        px=decomposition.px_after,
        py=decomposition.py,
        income=decomposition.income,
        color=final_budget_color,
        linewidth=final_budget_linewidth,
        linestyle=final_budget_linestyle,
        label=None,
        fill=False,
        fill_alpha=0.0,
    )
    _retag_last_budget(ax, "final_budget")

    points = [
        ("A", decomposition.A),
        ("B", decomposition.B),
        ("C", decomposition.C),
    ]
    for _, eq in points:
        plot_point(
            ax,
            x=eq.x,
            y=eq.y,
            color=point_color,
            markersize=point_markersize,
            marker="o",
            linestyle="None",
            zorder=7,
            role="bundle",
        )
    label_tol = _label_overlap_tolerance(ax)
    for labels, x, y in _group_overlapping_labels(points, tol=label_tol):
        annotate_math(
            ax,
            x=x,
            y=y,
            text=" = ".join(labels),
            color=point_color,
            offset=(6, 6),
            fontsize=12,
            zorder=8,
            role="bundle_label",
            default=Label(position=LabelPosition.TOP_RIGHT, offset=6),
        )

    if not show_arrows:
        return

    if arrows_below_axis or show_x_projections:
        _draw_x_projections(
            ax,
            decomposition=decomposition,
            substitution_color=substitution_color,
            income_color=income_color,
            linewidth=max(0.8, effect_arrow_linewidth * 0.7),
            substitution_effect=substitution_effect,
            income_effect=income_effect,
            effect_label=effect_label,
        )
        return

    _draw_effect_arrow(
        ax,
        start=(decomposition.A.x, decomposition.A.y),
        end=(decomposition.B.x, decomposition.B.y),
        color=substitution_color,
        linewidth=effect_arrow_linewidth,
        role="substitution",
        opacity=_opacity(substitution_effect),
    )
    _draw_effect_arrow(
        ax,
        start=(decomposition.B.x, decomposition.B.y),
        end=(decomposition.C.x, decomposition.C.y),
        color=income_color,
        linewidth=effect_arrow_linewidth,
        role="income",
        opacity=_opacity(income_effect),
    )
    a, b, c = ((eq.x, eq.y) for eq in (decomposition.A, decomposition.B, decomposition.C))
    _draw_effect_label(ax, start=a, end=b, effect=substitution_effect, color=substitution_color,
                       transform=ax.transData, role="substitution_label", default=effect_label)
    _draw_effect_label(ax, start=b, end=c, effect=income_effect, color=income_color,
                       transform=ax.transData, role="income_label", default=effect_label)


def _draw_effect_arrow(
    ax,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float,
    role: str,
    opacity: float | None = None,
) -> None:
    arrow = ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "alpha": opacity,
        },
        zorder=8,
    )
    arrow._ev_role = role


def _draw_x_projections(
    ax,
    *,
    decomposition,
    substitution_color: str,
    income_color: str,
    linewidth: float,
    substitution_effect: Effect | None = None,
    income_effect: Effect | None = None,
    effect_label: Label | None = None,
) -> None:
    a_x = decomposition.A.x
    b_x = decomposition.B.x
    c_x = decomposition.C.x
    fig = ax.figure
    if fig.subplotpars.bottom < 0.22:
        fig.subplots_adjust(bottom=0.22)

    xaxis_t = ax.get_xaxis_transform()
    sub_y = _range_y(substitution_effect, SUBSTITUTION_RANGE_Y)
    inc_y = _range_y(income_effect, INCOME_RANGE_Y)
    # Guides reach just past the lowest range arrow.
    projection_bottom = min(sub_y, inc_y) - 0.01

    for eq in (decomposition.A, decomposition.B, decomposition.C):
        (projection,) = ax.plot(
            [eq.x, eq.x],
            [0.0, eq.y],
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            zorder=5,
        )
        projection._ev_role = "projection"
        (guide,) = ax.plot(
            [eq.x, eq.x],
            [0.0, projection_bottom],
            transform=xaxis_t,
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            zorder=6,
            clip_on=False,
        )
        guide._ev_role = "guide"

    x0, x1 = ax.get_xlim()
    # A zero effect has no range; an arrow there would be a bare head.
    min_length = 1e-3 * abs(x1 - x0)
    ranges = (
        (a_x, b_x, sub_y, substitution_color, _opacity(substitution_effect)),
        (b_x, c_x, inc_y, income_color, _opacity(income_effect)),
    )
    for (start_x, end_x, y, color, opacity) in ranges:
        if abs(end_x - start_x) <= min_length:
            continue
        effect_range = ax.annotate(
            "",
            xy=(end_x, y),
            xytext=(start_x, y),
            xycoords=xaxis_t,
            textcoords=xaxis_t,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": linewidth,
                "linestyle": "--",
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "alpha": opacity,
            },
            zorder=9,
            clip_on=False,
        )
        effect_range._ev_role = "range"
    _draw_effect_label(ax, start=(a_x, sub_y), end=(b_x, sub_y), effect=substitution_effect,
                       color=substitution_color, transform=xaxis_t, role="substitution_label", beyond_ends=True,
                       default=effect_label)
    _draw_effect_label(ax, start=(b_x, inc_y), end=(c_x, inc_y), effect=income_effect,
                       color=income_color, transform=xaxis_t, role="income_label", beyond_ends=True,
                       default=effect_label)


def _opacity(effect: Effect | None) -> float | None:
    return effect.opacity if effect is not None else None


def _range_y(effect: Effect | None, default: float) -> float:
    return effect.y if effect is not None and effect.y is not None else default


def _draw_effect_label(
    ax, *, start, end, effect: Effect | None, color: str, transform, role: str, beyond_ends: bool = False,
    default: Label | None = None,
) -> None:
    """Write ``effect.label`` beside the middle of an effect arrow.

    With *beyond_ends* (horizontal range arrows), left and right labels sit past
    the arrow's left or right end instead.
    """
    if effect is None or not effect.label:
        return
    text, style = effect.resolved_label(default or Label(fontsize=10))
    if not text:
        return
    (dx, dy), ha, va = placement(style.position, style.offset)
    if beyond_ends and style.position is LabelPosition.LEFT:
        anchor = min(start, end, key=lambda p: p[0])
    elif beyond_ends and style.position is LabelPosition.RIGHT:
        anchor = max(start, end, key=lambda p: p[0])
    else:
        anchor = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    label = ax.annotate(
        text,
        xy=anchor,
        xycoords=transform,
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=style.color or effect.color or color,
        fontsize=style.fontsize,
        zorder=9,
        annotation_clip=False,
    )
    label.set_visible(style.visible is not False)
    label.set_alpha(style.opacity if style.opacity is not None else effect.opacity)
    label._ev_role = role


def _retag_last_budget(ax, role: str) -> None:
    """Give the budget line just drawn its decomposition role."""
    line = next(line for line in reversed(ax.lines) if getattr(line, "_ev_role", None) == "budget")
    line._ev_role = role


def _label_overlap_tolerance(ax) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale = max(abs(x1 - x0), abs(y1 - y0), 1.0)
    return 0.015 * scale


def _group_overlapping_labels(
    points: Iterable[tuple[str, object]],
    *,
    tol: float,
) -> list[tuple[list[str], float, float]]:
    entries = [
        (label, float(eq.x), float(eq.y))
        for label, eq in points
    ]
    order = {"A": 0, "B": 1, "C": 2}
    groups: list[dict[str, object]] = []

    for label, x, y in entries:
        attached = False
        for group in groups:
            gx = group["x"]
            gy = group["y"]
            if abs(x - gx) <= tol and abs(y - gy) <= tol:
                group["labels"].append(label)
                pts = group["points"]
                pts.append((x, y))
                n = len(pts)
                group["x"] = sum(px for px, _ in pts) / n
                group["y"] = sum(py for _, py in pts) / n
                attached = True
                break
        if not attached:
            groups.append({
                "labels": [label],
                "points": [(x, y)],
                "x": x,
                "y": y,
            })

    merged: list[tuple[list[str], float, float]] = []
    for group in groups:
        labels = sorted(group["labels"], key=lambda token: order.get(token, 99))
        merged.append((labels, float(group["x"]), float(group["y"])))
    return merged

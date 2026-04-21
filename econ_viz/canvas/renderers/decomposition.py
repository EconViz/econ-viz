"""Renderer for price-effect decomposition overlays."""

from __future__ import annotations

from typing import Iterable

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
        )
        return

    _draw_effect_arrow(
        ax,
        start=(decomposition.A.x, decomposition.A.y),
        end=(decomposition.B.x, decomposition.B.y),
        color=substitution_color,
        linewidth=effect_arrow_linewidth,
    )
    _draw_effect_arrow(
        ax,
        start=(decomposition.B.x, decomposition.B.y),
        end=(decomposition.C.x, decomposition.C.y),
        color=income_color,
        linewidth=effect_arrow_linewidth,
    )


def _draw_effect_arrow(
    ax,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=8,
    )


def _draw_x_projections(
    ax,
    *,
    decomposition,
    substitution_color: str,
    income_color: str,
    linewidth: float,
) -> None:
    a_x = decomposition.A.x
    b_x = decomposition.B.x
    c_x = decomposition.C.x
    fig = ax.figure
    if fig.subplotpars.bottom < 0.22:
        fig.subplots_adjust(bottom=0.22)

    xaxis_t = ax.get_xaxis_transform()
    projection_bottom = -0.16

    for eq in (decomposition.A, decomposition.B, decomposition.C):
        ax.plot(
            [eq.x, eq.x],
            [0.0, eq.y],
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            zorder=5,
        )
        ax.plot(
            [eq.x, eq.x],
            [0.0, projection_bottom],
            transform=xaxis_t,
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            zorder=6,
            clip_on=False,
        )

    sub_y = -0.10
    inc_y = -0.15
    ax.annotate(
        "",
        xy=(b_x, sub_y),
        xytext=(a_x, sub_y),
        xycoords=xaxis_t,
        textcoords=xaxis_t,
        arrowprops={
            "arrowstyle": "<->",
            "color": substitution_color,
            "linewidth": linewidth,
            "linestyle": "--",
        },
        zorder=9,
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(c_x, inc_y),
        xytext=(b_x, inc_y),
        xycoords=xaxis_t,
        textcoords=xaxis_t,
        arrowprops={
            "arrowstyle": "<->",
            "color": income_color,
            "linewidth": linewidth,
            "linestyle": "--",
        },
        zorder=9,
        clip_on=False,
    )


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

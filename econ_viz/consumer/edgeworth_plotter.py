"""Plotting helpers for Edgeworth diagrams."""

from __future__ import annotations

import numpy as np

from ..canvas.stroke import tag, tag_attr
from ..enums import LabelPosition
from ..themes.label import Label


def plot_endowment(
    ax,
    *,
    x: float,
    y: float,
    total_x: float,
    total_y: float,
    color: str,
    markersize: float,
    label: str,
) -> None:
    """Draw and label the endowment point."""
    (point,) = ax.plot(
        x,
        y,
        "o",
        color=color,
        markersize=markersize,
        zorder=20,
    )
    tag(point, "endowment")
    text = ax.annotate(
        rf"${label}$",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        color=color,
        fontsize=11,
        zorder=21,
    )
    tag(text, "endowment_label")
    tag_attr(text, "_ev_label_default", Label(position=LabelPosition.TOP_RIGHT, offset=5))


def plot_price_line(
    ax,
    *,
    points: list[tuple[float, float]],
    color: str,
    linewidth: float,
    linestyle: str,
    label: str,
) -> None:
    """Draw the budget/price line segment inside the Edgeworth box."""
    if len(points) < 2:
        return
    (line,) = ax.plot(
        [points[0][0], points[-1][0]],
        [points[0][1], points[-1][1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    tag(line, "price")


def plot_equilibrium_marker(
    ax,
    *,
    x: float,
    y: float,
    total_x: float,
    total_y: float,
    color: str,
    marker: str,
    markersize: float,
    label: str,
) -> None:
    """Draw and label Walrasian equilibrium marker."""
    (point,) = ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        markersize=markersize,
        label=rf"${label}$",
        zorder=22,
    )
    tag(point, "walrasian")
    text = ax.annotate(
        rf"${label}$",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        color=color,
        fontsize=11,
        zorder=23,
    )
    tag(text, "walrasian_label")
    tag_attr(text, "_ev_label_default", Label(position=LabelPosition.TOP_RIGHT, offset=5))


def plot_indifference_pair(
    ax,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    U_a: np.ndarray,
    U_b: np.ndarray,
    levels_a: list[float],
    levels_b: list[float],
    color_a: str,
    color_b: str,
    linewidth: float,
    linestyle_a: str = "-",
    linestyle_b: str = "-",
) -> None:
    """Draw both agents' indifference contour families."""
    cs = ax.contour(
        X,
        Y,
        U_a,
        levels=levels_a,
        colors=color_a,
        linewidths=linewidth,
        linestyles=linestyle_a,
    )
    tag(cs, "curve_a")
    cs = ax.contour(
        X,
        Y,
        U_b,
        levels=levels_b,
        colors=color_b,
        linewidths=linewidth,
        linestyles=linestyle_b,
    )
    tag(cs, "curve_b")


def plot_contract_curve(
    ax,
    *,
    points: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str,
    label: str,
) -> None:
    """Draw contract curve polyline if points exist."""
    if len(points) == 0:
        return
    (line,) = ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    tag(line, "contract")


def plot_core(
    ax,
    *,
    core_points: np.ndarray,
    color: str,
    linewidth: float,
    label: str,
    min_points: int = 2,
) -> None:
    """Draw the core as a segment or singleton point."""
    if len(core_points) >= min_points:
        (line,) = ax.plot(core_points[:, 0], core_points[:, 1], color=color, linewidth=linewidth, label=label)
        tag(line, "core")
    elif len(core_points) == 1:
        (point,) = ax.plot(core_points[0, 0], core_points[0, 1], "o", color=color, label=label)
        tag(point, "core_point")

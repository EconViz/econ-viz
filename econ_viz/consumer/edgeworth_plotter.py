"""Plotting helpers for Edgeworth diagrams."""

from __future__ import annotations

import numpy as np


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
    ax.plot(
        x,
        y,
        "o",
        color=color,
        markersize=markersize,
        zorder=20,
    )
    ax.text(
        x + total_x * 0.015,
        y + total_y * 0.015,
        rf"${label}$",
        color=color,
        fontsize=11,
        zorder=21,
    )


def plot_price_line(ax, *, points: list[tuple[float, float]], color: str, linewidth: float, label: str) -> None:
    """Draw the budget/price line segment inside the Edgeworth box."""
    if len(points) < 2:
        return
    ax.plot(
        [points[0][0], points[-1][0]],
        [points[0][1], points[-1][1]],
        color=color,
        linewidth=linewidth,
        label=label,
    )


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
    ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        markersize=markersize,
        label=rf"${label}$",
        zorder=22,
    )
    ax.text(
        x + total_x * 0.012,
        y + total_y * 0.012,
        rf"${label}$",
        color=color,
        fontsize=11,
        zorder=23,
    )


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
) -> None:
    """Draw both agents' indifference contour families."""
    ax.contour(X, Y, U_a, levels=levels_a, colors=color_a, linewidths=linewidth)
    ax.contour(X, Y, U_b, levels=levels_b, colors=color_b, linewidths=linewidth, linestyles="--")


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
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


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
        ax.plot(core_points[:, 0], core_points[:, 1], color=color, linewidth=linewidth, label=label)
    elif len(core_points) == 1:
        ax.plot(core_points[0, 0], core_points[0, 1], "o", color=color, label=label)

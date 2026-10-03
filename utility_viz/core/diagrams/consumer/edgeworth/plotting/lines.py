"""Line plots (price line, contract curve, core) for Edgeworth diagrams."""

from __future__ import annotations

import numpy as np

from utility_viz.core.rendering.stroke import tag


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

"""Indifference contour families for Edgeworth diagrams."""

from __future__ import annotations

import numpy as np

from utility_viz.core.rendering.stroke import tag


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

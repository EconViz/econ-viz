"""Low-level plotting primitives used by canvas renderers."""

from __future__ import annotations


def plot_point(
    ax,
    *,
    x: float,
    y: float,
    color: str,
    markersize: float,
    marker: str = "o",
    linestyle: str = "None",
    zorder: int = 6,
    clip_on: bool = False,
) -> None:
    """Draw a point marker with consistent defaults."""
    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        markersize=markersize,
        clip_on=clip_on,
        zorder=zorder,
    )


def annotate_math(
    ax,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    offset: tuple[float, float] = (5, 5),
    fontsize: int = 12,
    zorder: int = 7,
) -> None:
    """Add a math-formatted annotation offset from a point."""
    ax.annotate(
        rf"${text}$",
        (x, y),
        textcoords="offset points",
        xytext=offset,
        fontsize=fontsize,
        color=color,
        zorder=zorder,
    )


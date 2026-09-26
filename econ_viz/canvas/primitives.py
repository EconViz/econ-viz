"""Low-level plotting primitives used by canvas renderers."""

from __future__ import annotations

from ..themes.label import Label
from .stroke import tag, tag_attr


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
    role: str,
):
    """Draw a point marker with consistent defaults and return its line."""
    (point,) = ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        markersize=markersize,
        clip_on=clip_on,
        zorder=zorder,
    )
    tag(point, role)
    return point


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
    role: str,
    default: Label | None = None,
):
    """Add a math-formatted annotation offset from a point and return it."""
    text_artist = ax.annotate(
        rf"${text}$",
        (x, y),
        textcoords="offset points",
        xytext=offset,
        fontsize=fontsize,
        color=color,
        zorder=zorder,
    )
    tag(text_artist, role)
    tag_attr(text_artist, "_ev_label_default", default)
    return text_artist

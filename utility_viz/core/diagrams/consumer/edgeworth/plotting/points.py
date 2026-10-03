"""Point markers (endowment, Walrasian equilibrium) for Edgeworth diagrams."""

from __future__ import annotations

from utility_viz.core.rendering.stroke import tag, tag_attr
from utility_viz.core.styles.label import Label
from utility_viz.enums import LabelPosition


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

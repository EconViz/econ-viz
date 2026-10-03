"""Base styling of a :class:`~utility_viz.core.canvas.base.Canvas`: limits, labels, spines, arrows, background."""

from __future__ import annotations

from typing import Any, cast

from matplotlib.patches import FancyArrowPatch

from utility_viz.core.canvas._labels import _X_LABEL_POSITIONS, _Y_LABEL_POSITIONS, _label_math, _math_wrap
from utility_viz.core.canvas._state import CanvasState
from utility_viz.core.constants.canvas import ARROW_HEAD_ONLY_FRAC, ARROW_WEDGE_FRAC
from utility_viz.core.rendering.stroke import tag, tag_attr
from utility_viz.enums import ArrowStyle, LineStyle


def apply_base_style(canvas: CanvasState) -> None:
    """Configure axes to match textbook economic diagram conventions."""
    canvas.ax.set_xlim(0, canvas.x_max)
    canvas.ax.set_ylim(0, canvas.y_max)

    # Turn off tick labels
    canvas.ax.set_xticklabels([])
    canvas.ax.set_yticklabels([])
    canvas.ax.tick_params(length=0)

    _add_axis_labels(canvas)
    _add_origin_label(canvas)
    _add_title(canvas)
    _style_spines(canvas)
    _add_axis_arrows(canvas)
    _style_background(canvas)


def _add_axis_labels(canvas: CanvasState) -> None:
    """Place the x and y labels at the axis tips."""
    for axis, text, position, style, tip, layout in (
        ("x", canvas.x_label, canvas.x_label_pos, canvas.x_label_style, (canvas.x_max, 0), _X_LABEL_POSITIONS),
        ("y", canvas.y_label, canvas.y_label_pos, canvas.y_label_style, (0, canvas.y_max), _Y_LABEL_POSITIONS),
    ):
        (dx, dy), ha, va = layout[position]
        if style.offset is not None:
            # Layout offsets are 8 pt along one direction; rescale to the requested distance.
            dx, dy = dx / 8 * style.offset, dy / 8 * style.offset
        axis_text = canvas.ax.annotate(
            _label_math(text),
            xy=tip,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=style.fontsize,
            color=style.color or canvas.theme.label_color,
            clip_on=False,
        )
        axis_text.set_visible(style.visible is not False)
        axis_text.set_alpha(style.opacity)
        tag_attr(axis_text, "_ev_axis_label", axis)


def _add_origin_label(canvas: CanvasState) -> None:
    """Place the ``0`` marker just outside the origin."""
    origin = canvas.ax.text(
        -canvas.x_max * 0.03,
        -canvas.y_max * 0.03,
        _label_math(canvas.origin_text),
        ha="right",
        va="top",
        fontsize=canvas.origin_style.fontsize,
        color=canvas.origin_style.color or canvas.theme.label_color,
    )
    origin.set_visible(canvas.origin_style.visible is not False)
    origin.set_alpha(canvas.origin_style.opacity)
    tag(origin, "origin_label")


def _add_title(canvas: CanvasState) -> None:
    """Set the axes title, when the canvas has one."""
    if not canvas.title:
        return
    style = canvas.title_style
    title_kwargs: dict[str, Any] = {"fontsize": style.fontsize} if style.fontsize else {}
    title = canvas.ax.set_title(
        _math_wrap(canvas.title),
        color=style.color or canvas.theme.label_color,
        pad=18,
        **title_kwargs,
    )
    title.set_visible(style.visible is not False)
    title.set_alpha(style.opacity)


def _style_spines(canvas: CanvasState) -> None:
    """Hide the top/right spines and style the bottom/left ones from the axis strokes."""
    canvas.ax.spines["top"].set_visible(False)
    canvas.ax.spines["right"].set_visible(False)
    for spine, stroke in (("bottom", canvas.x_axis_stroke), ("left", canvas.y_axis_stroke)):
        canvas.ax.spines[spine].set_color(stroke.color)
        canvas.ax.spines[spine].set_linewidth(stroke.width)
        # Theme.__post_init__ / Stroke.__post_init__ always normalise style to a LineStyle.
        canvas.ax.spines[spine].set_linestyle(cast(LineStyle, stroke.style).value)
        canvas.ax.spines[spine].set_alpha(stroke.opacity)


def _arrow_frac(style: ArrowStyle) -> float:
    return ARROW_WEDGE_FRAC if style is ArrowStyle.WEDGE else ARROW_HEAD_ONLY_FRAC


def _add_axis_arrows(canvas: CanvasState) -> None:
    """Draw the arrow terminators at the axis tips."""
    for axis, stroke, tip in (
        ("x", canvas.x_axis_stroke, (canvas.x_max, 0)),
        ("y", canvas.y_axis_stroke, (0, canvas.y_max)),
    ):
        if stroke.arrow is None:
            continue
        arrow_style = cast(ArrowStyle, stroke.arrow)
        frac = _arrow_frac(arrow_style)
        start = (canvas.x_max * (1 - frac), 0) if axis == "x" else (0, canvas.y_max * (1 - frac))
        arrow = FancyArrowPatch(
            start,
            tip,
            arrowstyle=arrow_style.value,
            mutation_scale=12,
            linewidth=stroke.width,
            color=stroke.color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            alpha=stroke.opacity,
        )
        tag_attr(arrow, "_ev_axis_arrow", axis)
        tag_attr(arrow, "_ev_arrow_style", stroke.arrow)
        canvas.ax.add_patch(arrow)


def _style_background(canvas: CanvasState) -> None:
    """Transparent by default, or ``theme.background_color`` when set."""
    color = canvas.theme.background_color
    if color is not None:
        canvas.fig.patch.set_facecolor(color)
        canvas.fig.patch.set_alpha(1.0)
        canvas.ax.patch.set_facecolor(color)
        canvas.ax.patch.set_alpha(1.0)
    else:
        canvas.fig.patch.set_alpha(0.0)
        canvas.ax.patch.set_alpha(0.0)

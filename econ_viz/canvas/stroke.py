"""Per-line styling: width, line style, colour, and an optional arrowhead."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager

import numpy as np
from matplotlib.collections import Collection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation

from ..constants.canvas import ARROW_HEAD_ONLY_FRAC, ARROW_WEDGE_FRAC
from ..enums import ArrowStyle
from ..themes.stroke import Stroke


def tag(artist, role: str):
    """Mark *artist* with the line role that a Stroke can target."""
    artist._ev_role = role
    return artist


def _children(ax) -> set[int]:
    return {id(a) for a in ax.get_children()}


@contextmanager
def styled(canvas, strokes: Mapping[str, Stroke | None], default_role: str | None = None):
    """Apply *strokes* (role → Stroke) to lines and legend entries drawn inside the block.

    Lines created without a role get *default_role*.
    """
    ax = canvas.ax
    before = _children(ax)
    handles = getattr(canvas, "_legend_handles", [])
    handles_before = len(handles)
    yield
    new = [a for a in ax.get_children() if id(a) not in before]
    if default_role:
        for artist in new:
            if getattr(artist, "_ev_role", None) is None and _is_line(artist):
                tag(artist, default_role)
    apply_strokes(ax, new, strokes)
    for handle in handles[handles_before:]:
        stroke = strokes.get(getattr(handle, "_ev_role", default_role))
        if stroke is not None and isinstance(handle, Line2D):
            _style_line(handle, stroke)


def apply_strokes(ax, artists: Iterable, strokes: Mapping[str, Stroke | None]) -> None:
    for artist in artists:
        stroke = strokes.get(getattr(artist, "_ev_role", None))
        if stroke is None:
            continue
        if isinstance(artist, Annotation) and artist.arrow_patch is not None:
            _style_patch(artist.arrow_patch, stroke)
            if stroke.arrow is ArrowStyle.WEDGE and artist.xycoords == artist.anncoords:
                # The annotation keeps a plain shaft; the wedge covers only its tip, as on other lines.
                add_arrowhead(
                    ax, [artist.xyann, artist.xy], ArrowStyle.WEDGE,
                    color=artist.arrow_patch.get_edgecolor(), width=artist.arrow_patch.get_linewidth(),
                    transform=ax.transData if artist.xycoords == "data" else artist.xycoords,
                    role=artist._ev_role,
                )
        elif isinstance(artist, Line2D) and _is_line(artist):
            _style_line(artist, stroke)
            if stroke.arrow:
                add_arrowhead(
                    ax, artist.get_xydata(), stroke.arrow,
                    color=artist.get_color(), width=artist.get_linewidth(),
                    transform=artist.get_transform(), role=artist._ev_role,
                )
        elif isinstance(artist, Collection) and not isinstance(artist, PolyCollection):
            _style_collection(artist, stroke)
            if stroke.arrow:
                color, width = artist.get_edgecolor()[0], artist.get_linewidth()[0]
                for path in artist.get_paths():
                    for segment in path.to_polygons(closed_only=False):
                        add_arrowhead(
                            ax, segment, stroke.arrow, color=color, width=width,
                            transform=artist.get_transform(), role=artist._ev_role,
                        )


def _is_line(artist) -> bool:
    if isinstance(artist, Line2D):
        return artist.get_linestyle() not in ("None", "", " ")
    return isinstance(artist, Collection) and not isinstance(artist, PolyCollection)


def _style_line(line: Line2D, stroke: Stroke) -> None:
    if stroke.width is not None:
        line.set_linewidth(stroke.width)
    if stroke.style is not None:
        line.set_linestyle(stroke.style.value)
    if stroke.color is not None:
        line.set_color(stroke.color)


def _style_collection(collection: Collection, stroke: Stroke) -> None:
    if stroke.width is not None:
        collection.set_linewidth(stroke.width)
    if stroke.style is not None:
        collection.set_linestyle(stroke.style.value)
    if stroke.color is not None:
        collection.set_edgecolor(stroke.color)


def _style_patch(patch, stroke: Stroke) -> None:
    if stroke.width is not None:
        patch.set_linewidth(stroke.width)
    if stroke.style is not None:
        patch.set_linestyle(stroke.style.value)
    if stroke.color is not None:
        patch.set_color(stroke.color)
    if stroke.arrow is not None:
        # A wedge along a whole annotation becomes a long taper, so its head is drawn separately.
        patch.set_arrowstyle("-" if stroke.arrow is ArrowStyle.WEDGE else stroke.arrow.value)


def add_arrowhead(
    ax, points, style: ArrowStyle, *, color, width: float, transform, role: str | None = None
) -> FancyArrowPatch | None:
    """Draw an arrowhead at the last point of *points* (N×2), pointing along the line."""
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    end = points[-1]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(steps.sum())
    if total == 0:
        return None
    if style is ArrowStyle.WEDGE:
        # Same on-screen size on every line: a share of the axes' longer side, capped at the line.
        display = transform.transform(points)
        display_total = float(np.linalg.norm(np.diff(display, axis=0), axis=1).sum())
        bbox = ax.get_window_extent()
        wanted = ARROW_WEDGE_FRAC * max(bbox.width, bbox.height)
        remaining = total * min(wanted / display_total, 1.0) if display_total > 0 else 0.0
    else:
        remaining = ARROW_HEAD_ONLY_FRAC * total
    # Walk back along the line until the requested length is covered.
    start = points[-2]
    for i in range(len(points) - 1, 0, -1):
        if steps[i - 1] >= remaining:
            start = points[i] + (points[i - 1] - points[i]) * (remaining / steps[i - 1])
            break
        remaining -= steps[i - 1]
        start = points[i - 1]
    arrow = FancyArrowPatch(
        tuple(start),
        tuple(end),
        arrowstyle=style.value,
        mutation_scale=12,
        linewidth=width,
        color=color,
        shrinkA=0,
        shrinkB=0,
        transform=transform,
        clip_on=False,
        zorder=6,
    )
    arrow._ev_arrow_for = role
    arrow._ev_arrow_style = style
    ax.add_patch(arrow)
    return arrow

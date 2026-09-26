"""Per-line styling: width, line style, colour, and an optional arrowhead."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import cast

import numpy as np
from matplotlib.collections import Collection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation, Text

from ..constants.canvas import ARROW_HEAD_ONLY_FRAC, ARROW_WEDGE_FRAC
from ..enums import ArrowStyle, LabelPosition, LineStyle
from ..themes.label import Label
from ..themes.marker import Marker
from ..themes.stroke import Stroke
from .labels import placement


def tag(artist, role: str):
    """Mark *artist* with the line role that a Stroke can target."""
    artist._ev_role = role
    return artist


def tag_attr(artist, name: str, value):
    """Set an ``_ev_*`` attribute other than role (e.g. label defaults, arrow metadata)."""
    setattr(artist, name, value)
    return artist


def _children(ax) -> set[int]:
    return {id(a) for a in ax.get_children()}


@contextmanager
def styled(
    canvas,
    strokes: Mapping[str, Stroke | None],
    default_role: str | None = None,
    markers: Mapping[str, Marker | None] | None = None,
    labels: Mapping[str, Label | None] | None = None,
):
    """Apply *strokes* and *markers* (role → style) to what is drawn inside the block.

    Lines created without a role get *default_role*. Legend entries follow.
    """
    ax = canvas.ax
    strokes, markers = _with_config(canvas, strokes, markers)
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
    markers = markers or {}
    apply_markers(new, markers)
    apply_labels(new, labels or {}, markers)
    for handle in handles[handles_before:]:
        role: str | None = getattr(handle, "_ev_role", default_role)
        stroke = strokes.get(role) if role is not None else None
        if stroke is not None and isinstance(handle, Line2D):
            _style_line(handle, stroke)
        apply_markers([handle], markers)


# Drawing role -> Theme property that a config file may override.
_ROLE_STROKES = {
    "budget": "budget_stroke",
    "original_budget": "budget_stroke",
    "curve": "ic_stroke",
    "secondary_curve": "secondary_ic_stroke",
    "ray": "ray_stroke",
    "path": "path_stroke",
    "drop": "drop_stroke",
    "compensated_budget": "compensated_budget_stroke",
    "final_budget": "final_budget_stroke",
    "substitution": "substitution_stroke",
    "income": "income_stroke",
    "projection": "projection_stroke",
    "guide": "guide_stroke",
    "subsistence": "subsistence_stroke",
    "contract": "contract_stroke",
    "core": "core_stroke",
    "price": "price_stroke",
}
_ROLE_MARKERS = {
    "equilibrium": "eq_marker",
    "bundle": "eq_marker",
    "point": "point_marker",
    "kink": "kink_marker",
    "bliss": "bliss_marker",
    "path_point": "path_marker",
    "core_point": "core_marker",
    "endowment": "endowment_marker",
    "walrasian": "walrasian_marker",
}


def _with_config(canvas, strokes, markers):
    """Add the strokes and markers a config file set, under the explicit ones."""
    overrides = getattr(getattr(canvas, "theme", None), "_ev_overrides", None)
    if not overrides:
        return strokes, markers
    strokes, markers = dict(strokes), dict(markers or {})
    for table, roles in ((strokes, _ROLE_STROKES), (markers, _ROLE_MARKERS)):
        for role, name in roles.items():
            configured = overrides.get(name)
            if configured is not None:
                explicit = table.get(role)
                table[role] = explicit.merged_over(configured) if explicit is not None else configured
    return strokes, markers


def apply_strokes(ax, artists: Iterable, strokes: Mapping[str, Stroke | None]) -> None:
    for artist in artists:
        role: str | None = getattr(artist, "_ev_role", None)
        if role is None:
            continue
        stroke = strokes.get(role)
        if stroke is None:
            continue
        if isinstance(artist, Annotation) and artist.arrow_patch is not None:
            _style_patch(artist.arrow_patch, stroke)
            if stroke.arrow is ArrowStyle.WEDGE and artist.xycoords == artist.anncoords:
                # The annotation keeps a plain shaft; the wedge covers only its tip, as on other lines.
                add_arrowhead(
                    ax,
                    [artist.xyann, artist.xy],
                    ArrowStyle.WEDGE,
                    color=artist.arrow_patch.get_edgecolor(),
                    width=artist.arrow_patch.get_linewidth(),
                    transform=ax.transData if artist.xycoords == "data" else artist.xycoords,
                    role=role,
                    opacity=stroke.opacity,
                )
        elif isinstance(artist, Line2D) and _is_line(artist):
            _style_line(artist, stroke)
            if stroke.arrow:
                add_arrowhead(
                    ax,
                    artist.get_xydata(),
                    cast(ArrowStyle, stroke.arrow),
                    color=artist.get_color(),
                    width=artist.get_linewidth(),
                    transform=artist.get_transform(),
                    role=role,
                    opacity=stroke.opacity,
                )
        elif isinstance(artist, Collection) and not isinstance(artist, PolyCollection):
            _style_collection(artist, stroke)
            if stroke.arrow:
                color = artist.get_edgecolor()[0]
                width = float(np.ravel(artist.get_linewidth())[0])
                for path in artist.get_paths():
                    for segment in path.to_polygons(closed_only=False):
                        add_arrowhead(
                            ax,
                            segment,
                            cast(ArrowStyle, stroke.arrow),
                            color=color,
                            width=width,
                            transform=artist.get_transform(),
                            role=role,
                            opacity=stroke.opacity,
                        )


def apply_markers(artists: Iterable, markers: Mapping[str, Marker | None]) -> None:
    """Restyle marker-only lines whose role has a Marker."""
    for artist in artists:
        role: str | None = getattr(artist, "_ev_role", None)
        marker = markers.get(role) if role is not None else None
        if marker is None or not isinstance(artist, Line2D):
            continue
        if marker.color is not None:
            artist.set_color(marker.color)
            artist.set_markerfacecolor(marker.color)
            artist.set_markeredgecolor(marker.color)
        if marker.size is not None:
            artist.set_markersize(marker.size)
        if marker.shape is not None:
            artist.set_marker(marker.shape)
        if marker.opacity is not None:
            artist.set_alpha(marker.opacity)


def apply_labels(artists: Iterable, labels: Mapping[str, Label | None], markers: Mapping[str, Marker | None]) -> None:
    """Restyle text labels whose role (``<point role>_label``) has a Label or a coloured Marker."""
    for artist in artists:
        role = getattr(artist, "_ev_role", None)
        if not isinstance(role, str) or not role.endswith("_label") or not isinstance(artist, Text):
            continue
        label = labels.get(role) or Label()
        default = getattr(artist, "_ev_label_default", None) or Label()
        marker = markers.get(role[: -len("_label")])
        color = label.color or (marker.color if marker is not None else None)
        if color is not None:
            artist.set_color(color)
        if label.visible is False:
            artist.set_visible(False)
        if label.text is not None:
            artist.set_text(rf"${label.text}$")
        if label.fontsize is not None:
            artist.set_fontsize(label.fontsize)
        if label.opacity is not None:
            artist.set_alpha(label.opacity)
        placed = label.merged_over(default)
        moved = (placed.position, placed.offset) != (default.position, default.offset)
        if moved and isinstance(artist, Annotation):
            # Label.__post_init__ always normalises position to a LabelPosition (or None).
            position = cast("LabelPosition | None", placed.position)
            xytext, ha, va = placement(
                position or LabelPosition.TOP_RIGHT, placed.offset if placed.offset is not None else 5.0
            )
            artist.set_position(xytext)
            artist.set_horizontalalignment(ha)
            artist.set_verticalalignment(va)


def _is_line(artist) -> bool:
    if isinstance(artist, Line2D):
        return artist.get_linestyle() not in ("None", "", " ")
    return isinstance(artist, Collection) and not isinstance(artist, PolyCollection)


def _style_line(line: Line2D, stroke: Stroke) -> None:
    if stroke.width is not None:
        line.set_linewidth(stroke.width)
    if stroke.style is not None:
        # Stroke.__post_init__ always normalises style to a LineStyle.
        line.set_linestyle(cast(LineStyle, stroke.style).value)
    if stroke.color is not None:
        line.set_color(stroke.color)
    if stroke.opacity is not None:
        line.set_alpha(stroke.opacity)


def _style_collection(collection: Collection, stroke: Stroke) -> None:
    if stroke.width is not None:
        collection.set_linewidth(stroke.width)
    if stroke.style is not None:
        collection.set_linestyle(cast(LineStyle, stroke.style).value)
    if stroke.color is not None:
        collection.set_edgecolor(stroke.color)
    if stroke.opacity is not None:
        collection.set_alpha(stroke.opacity)


def _style_patch(patch, stroke: Stroke) -> None:
    if stroke.width is not None:
        patch.set_linewidth(stroke.width)
    if stroke.style is not None:
        patch.set_linestyle(cast(LineStyle, stroke.style).value)
    if stroke.color is not None:
        patch.set_color(stroke.color)
    if stroke.opacity is not None:
        patch.set_alpha(stroke.opacity)
    if stroke.arrow is not None:
        # A wedge along a whole annotation becomes a long taper, so its head is drawn separately.
        arrow = cast(ArrowStyle, stroke.arrow)
        patch.set_arrowstyle("-" if arrow is ArrowStyle.WEDGE else arrow.value)


def add_arrowhead(
    ax,
    points,
    style: ArrowStyle,
    *,
    color,
    width: float,
    transform,
    role: str | None = None,
    opacity: float | None = None,
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
        alpha=opacity,
    )
    tag_attr(arrow, "_ev_arrow_for", role)
    tag_attr(arrow, "_ev_arrow_style", style)
    ax.add_patch(arrow)
    return arrow

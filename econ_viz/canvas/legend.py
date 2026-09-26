"""Place a legend where it covers as little of the diagram as possible."""

from __future__ import annotations

import math

from matplotlib.collections import Collection
from matplotlib.legend import Legend as MplLegend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from ..enums import LegendPosition
from ..themes.legend import Legend

# Order tried by automatic placement; ties keep the earlier corner.
_CORNERS = (
    LegendPosition.UPPER_RIGHT,
    LegendPosition.UPPER_LEFT,
    LegendPosition.LOWER_RIGHT,
    LegendPosition.LOWER_LEFT,
)
# Outside positions: (legend loc, anchor point on the axes' tight bbox as fractions).
_OUTSIDE = {
    LegendPosition.TOP: ("lower center", (0.5, 1.0)),
    LegendPosition.BOTTOM: ("upper center", (0.5, 0.0)),
    LegendPosition.LEFT: ("center right", (0.0, 0.5)),
    LegendPosition.RIGHT: ("center left", (1.0, 0.5)),
}
_GAP_PX = 6.0


def place_legend(ax, handles: list, labels: list[str], legend: Legend) -> MplLegend | None:
    """Draw a legend on *ax* as *legend* says and return it (``None`` when hidden or empty)."""
    old = ax.get_legend()
    if old is not None:
        old.remove()
    if legend.visible is False or not handles:
        return None
    position = legend.position or LegendPosition.AUTO
    if position is LegendPosition.AUTO:
        position = _best_position(ax, handles, labels, legend)
    return _draw(ax, handles, labels, legend, position)


def _draw(ax, handles, labels, legend: Legend, position: LegendPosition) -> MplLegend:
    kwargs = {
        "handles": handles,
        "labels": labels,
        "frameon": bool(legend.frame),
        "fontsize": legend.fontsize,
        "ncols": legend.columns or _default_columns(position, len(handles)),
    }
    if position.inside:
        return ax.legend(loc=position.value, **kwargs)
    loc, (fx, fy) = _OUTSIDE[position]
    fig = ax.figure
    renderer = _renderer(fig)
    box = ax.get_tightbbox(renderer)
    dx = {0.0: -_GAP_PX, 1.0: _GAP_PX}.get(fx, 0.0)
    dy = {0.0: -_GAP_PX, 1.0: _GAP_PX}.get(fy, 0.0)
    anchor = (box.x0 + fx * box.width + dx, box.y0 + fy * box.height + dy)
    x, y = fig.transFigure.inverted().transform(anchor)
    return ax.legend(loc=loc, bbox_to_anchor=(x, y), bbox_transform=fig.transFigure, **kwargs)


def _renderer(fig):
    # Agg canvases expose get_renderer(); other backends go through the figure.
    get = getattr(fig.canvas, "get_renderer", None)
    return get() if get is not None else fig._get_renderer()


def _default_columns(position: LegendPosition, count: int) -> int:
    if position in (LegendPosition.TOP, LegendPosition.BOTTOM):
        return min(count, 3)
    return 1


def _best_position(ax, handles, labels, legend: Legend) -> LegendPosition:
    """Inside corner covering the fewest diagram elements; outside right when all cover some."""
    renderer = _renderer(ax.figure)
    obstacles = _obstacles(ax, renderer)
    best, best_score = None, math.inf
    for corner in _CORNERS:
        trial = _draw(ax, handles, labels, legend, corner)
        box = trial.get_window_extent(renderer).padded(2.0)
        trial.remove()
        score = sum(1 for hits in obstacles if hits(box))
        if score < best_score:
            best, best_score = corner, score
        if score == 0:
            break
    return best if best_score == 0 else LegendPosition.RIGHT


def _obstacles(ax, renderer) -> list:
    """Hit tests (display-space Bbox -> bool) for everything drawn on *ax*."""
    tests = []
    for artist in ax.get_children():
        if not artist.get_visible() or isinstance(artist, MplLegend):
            continue
        if isinstance(artist, Line2D):
            display = artist.get_transform().transform_path(artist.get_path())
            if artist.get_linestyle() in ("None", "", " ") or len(display.vertices) == 1:
                tests.append(_points_test(display.vertices))
            else:
                tests.append(_path_test(display))
        elif isinstance(artist, Collection):
            tests.extend(
                _path_test(artist.get_transform().transform_path(path))
                for path in artist.get_paths()
                if len(path.vertices)
            )
        elif isinstance(artist, Text):
            if not artist.get_text():
                arrow = getattr(artist, "arrow_patch", None)
                if arrow is not None and arrow.get_visible():
                    tests.append(_bbox_test(arrow.get_window_extent(renderer)))
                continue
            tests.append(_bbox_test(artist.get_window_extent(renderer)))
        elif isinstance(artist, Patch) and getattr(artist, "_ev_arrow_for", None) is not None:
            tests.append(_bbox_test(artist.get_window_extent(renderer)))
    return tests


def _path_test(display_path):
    return lambda box: display_path.intersects_bbox(box, filled=False)


def _points_test(points):
    def hits(box: Bbox) -> bool:
        return any(box.contains(float(x), float(y)) for x, y in points)
    return hits


def _bbox_test(extent: Bbox):
    return lambda box: box.overlaps(extent)

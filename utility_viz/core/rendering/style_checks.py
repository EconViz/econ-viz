"""Matplotlib-backed validation of colour and marker-shape strings used by styles."""

from __future__ import annotations

from matplotlib.colors import is_color_like
from matplotlib.markers import MarkerStyle

from utility_viz.core.styles.validation import register_checks


def _is_color(value: str) -> bool:
    return bool(is_color_like(value))


def _is_shape(value: str) -> bool:
    try:
        MarkerStyle(value)
    except ValueError:
        return False
    return True


register_checks(color=_is_color, shape=_is_shape)

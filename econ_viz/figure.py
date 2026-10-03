"""Deprecated alias of ``utility_viz.core.layout.figure`` (removed in 3.0.0)."""

from __future__ import annotations

from typing import Any

from econ_viz import _legacy


def __getattr__(name: str) -> Any:
    if name == "Figure":
        return _legacy.shim("Figure")
    import utility_viz.core.layout.figure as target

    try:
        return getattr(target, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

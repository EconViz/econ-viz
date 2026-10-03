"""Deprecated alias of ``utility_viz.core.animation`` (removed in 3.0.0)."""

from __future__ import annotations

from typing import Any

from econ_viz import _legacy

__path__ = []  # namespace-like: sub-paths resolve through the legacy finder


def __getattr__(name: str) -> Any:
    if name == "Animator":
        return _legacy.shim("Animator")
    import utility_viz.core.animation as target

    try:
        return getattr(target, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

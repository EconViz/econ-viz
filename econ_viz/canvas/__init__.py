"""Deprecated alias of ``utility_viz`` canvas names (removed in 3.0.0)."""

from __future__ import annotations

from typing import Any

from econ_viz import _legacy

__path__ = []  # namespace-like: sub-paths resolve through the legacy finder


def __getattr__(name: str) -> Any:
    if name in ("Canvas", "Figure"):
        return _legacy.shim(name)
    if name == "Layer":
        from utility_viz.models.curves import Layer

        return Layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Deprecated alias for the 1.x ``consumer`` package (removed in 3.0.0)."""

from __future__ import annotations

import importlib
from typing import Any

__path__ = []  # namespace-like: sub-paths resolve through the legacy finder

_DIAGRAMS = {"DemandDiagram", "EdgeworthBox", "EquilibriumFocusConfig"}
_MODELS = {"ConsumptionPath", "IncomePath", "LinearBudget", "PricePath", "EdgeworthState"}


def __getattr__(name: str) -> Any:
    if name in _DIAGRAMS:
        return getattr(importlib.import_module("utility_viz.core.diagrams.consumer"), name)
    if name in _MODELS:
        return getattr(importlib.import_module("utility_viz.models.consumer"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Backward-compatible re-exports for comparative statics helpers."""

from econ_viz.models.optimization.comparative import ComparativeStatics, comparative_statics
from econ_viz.models.optimization.slutsky import SlutskyMatrix, slutsky_matrix

__all__ = [
    "ComparativeStatics",
    "comparative_statics",
    "SlutskyMatrix",
    "slutsky_matrix",
]

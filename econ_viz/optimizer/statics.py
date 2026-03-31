"""Backward-compatible re-exports for comparative statics helpers."""

from .comparative import ComparativeStatics, comparative_statics
from .slutsky import SlutskyMatrix, slutsky_matrix

__all__ = [
    "ComparativeStatics",
    "comparative_statics",
    "SlutskyMatrix",
    "slutsky_matrix",
]

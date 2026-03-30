"""
econ_viz.enums — Categorical descriptors for economic model classification.

Enumerations defined here allow the plotting layer to adapt rendering
behaviour (e.g. drawing kink markers or expansion-path rays) based on
the qualitative shape of the underlying preference family, and to
validate export formats at save time.
"""

from .utility import UtilityType
from .extension import ExportFormat
from .returns import ReturnsToScale

__all__ = ["UtilityType", "ExportFormat", "ReturnsToScale"]

"""
econ_viz.enums — Categorical descriptors for economic model classification.

Enumerations defined here allow the plotting layer to adapt rendering
behaviour (e.g. drawing kink markers or expansion-path rays) based on
the qualitative shape of the underlying preference family, and to
validate export formats at save time.
"""

from .axis import ArrowStyle, LabelPosition, LineStyle
from .extension import ExportFormat
from .layout import Layout
from .legend import LegendPosition
from .returns import ReturnsToScale
from .utility import UtilityType

__all__ = [
    "ArrowStyle",
    "LabelPosition",
    "LineStyle",
    "UtilityType",
    "ExportFormat",
    "Layout",
    "LegendPosition",
    "ReturnsToScale",
]

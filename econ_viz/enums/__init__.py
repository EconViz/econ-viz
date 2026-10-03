"""
econ_viz.enums — Categorical descriptors for economic model classification.

Enumerations defined here allow the plotting layer to adapt rendering
behaviour (e.g. drawing kink markers or expansion-path rays) based on
the qualitative shape of the underlying preference family, and to
validate export formats at save time.
"""

from econ_viz.enums.axis import ArrowStyle, LabelPosition, LineStyle
from econ_viz.enums.extension import ExportFormat
from econ_viz.enums.layout import Layout
from econ_viz.enums.legend import LegendPosition
from econ_viz.enums.returns import ReturnsToScale
from econ_viz.enums.utility import UtilityType

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

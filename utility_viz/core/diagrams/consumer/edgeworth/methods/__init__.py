"""Public methods of :class:`EdgeworthBox`, grouped as stacked mixins."""

from utility_viz.core.diagrams.consumer.edgeworth.methods.indifference import IndifferenceMixin
from utility_viz.core.diagrams.consumer.edgeworth.methods.lines import LinesMixin
from utility_viz.core.diagrams.consumer.edgeworth.methods.points import PointsMixin

__all__ = ["IndifferenceMixin", "LinesMixin", "PointsMixin"]

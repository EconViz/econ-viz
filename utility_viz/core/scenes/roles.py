"""Concept-local theme roles for the utility-viz scene layers.

Roles are dotted: ``utility.budget.compensated`` inherits every field it does not set
from ``utility.budget`` (mosaickit's dotted fallback). Each member is a ``str``, so it
can be passed straight to a layer's ``role`` argument.
"""

from __future__ import annotations

from enum import Enum


class Budget(str, Enum):
    MAIN = "utility.budget"
    FILL = "utility.budget.fill"
    COMPENSATED = "utility.budget.compensated"


class Equilibrium(str, Enum):
    MAIN = "utility.equilibrium"
    LABEL = "utility.equilibrium.label"
    DROP = "utility.equilibrium.drop"
    RAY = "utility.equilibrium.ray"


class Indifference(str, Enum):
    MAIN = "utility.indifference"
    SECONDARY = "utility.indifference.secondary"
    LABEL = "utility.indifference.label"
    SECONDARY_LABEL = "utility.indifference.secondary.label"

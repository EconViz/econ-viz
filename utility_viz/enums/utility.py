"""
Enumeration of utility function shape categories.

Each member describes a qualitative property of the indifference map that
the rendering layer uses to decide which decorations (kink markers,
expansion-path rays, etc.) are appropriate.
"""

from enum import Enum, auto


class UtilityType(Enum):
    """Qualitative shape category of a utility function's indifference map.

    Members
    -------
    SMOOTH
        Indifference curves are everywhere differentiable (e.g. Cobb-Douglas, CES).
    KINKED
        Indifference curves exhibit non-differentiable kink points
        (e.g. Leontief / perfect complements).
    LINEAR
        Indifference curves are straight lines (e.g. perfect substitutes).
    """

    SMOOTH = auto()
    KINKED = auto()
    LINEAR = auto()

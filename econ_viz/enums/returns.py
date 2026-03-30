"""
Enumeration of returns-to-scale classifications.

Derived from the homogeneity degree *k* of a utility or production function
and used by :class:`~econ_viz.analysis.homogeneity.HomogeneityAnalyzer` to
summarise scaling behaviour in production-theoretic terms.
"""

from enum import Enum, auto


class ReturnsToScale(Enum):
    """Returns-to-scale classification derived from homogeneity degree.

    Members
    -------
    INCREASING
        Degree *k* > 1: doubling inputs more than doubles output.
    CONSTANT
        Degree *k* = 1: doubling inputs exactly doubles output.
    DECREASING
        0 < *k* < 1: doubling inputs less than doubles output.
    NOT_HOMOGENEOUS
        No consistent degree *k* exists across sampled bundles.
    """

    INCREASING = auto()
    CONSTANT = auto()
    DECREASING = auto()
    NOT_HOMOGENEOUS = auto()

    @classmethod
    def from_degree(cls, k: float | None, tol: float = 1e-4) -> "ReturnsToScale":
        """Classify a homogeneity degree into a returns-to-scale member.

        Parameters
        ----------
        k : float or None
            Estimated homogeneity degree, or ``None`` if not homogeneous.
        tol : float
            Tolerance for treating *k* as exactly 1 (constant returns).

        Returns
        -------
        ReturnsToScale
        """
        if k is None:
            return cls.NOT_HOMOGENEOUS
        if abs(k - 1.0) <= tol:
            return cls.CONSTANT
        if k > 1.0:
            return cls.INCREASING
        return cls.DECREASING

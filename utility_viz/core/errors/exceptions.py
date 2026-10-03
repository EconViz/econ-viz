"""
Custom exception hierarchy for utility-viz.

All package-level exceptions inherit from :class:`UtilityVizError`, allowing
callers to catch the entire family with a single ``except`` clause while
still being able to discriminate by subtype when needed.
"""


class UtilityVizError(Exception):
    """Base exception for the utility-viz package."""


#: Compatibility alias for the 1.x name; the same class object as :class:`UtilityVizError`.
EconVizError = UtilityVizError


class OptimizationError(UtilityVizError):
    """Raised when the equilibrium solver fails to converge or
    encounters an infeasible configuration (e.g. budget set is empty,
    utility is unbounded on the constraint)."""


class InvalidParameterError(UtilityVizError):
    """Raised when a model or canvas receives a parameter outside its
    valid domain (e.g. negative elasticity, rho=1 in CES)."""


class ExportError(UtilityVizError):
    """Raised when figure export fails (unsupported format, I/O error,
    or TikZ conversion issue)."""


class ParseError(UtilityVizError):
    """Raised when a LaTeX math string cannot be parsed into a valid
    utility function (e.g. unrecognised syntax, missing variables)."""

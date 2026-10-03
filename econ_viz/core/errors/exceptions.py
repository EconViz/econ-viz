"""
Custom exception hierarchy for econ-viz.

All package-level exceptions inherit from :class:`EconVizError`, allowing
callers to catch the entire family with a single ``except`` clause while
still being able to discriminate by subtype when needed.
"""


class EconVizError(Exception):
    """Base exception for the econ-viz package."""


class OptimizationError(EconVizError):
    """Raised when the equilibrium solver fails to converge or
    encounters an infeasible configuration (e.g. budget set is empty,
    utility is unbounded on the constraint)."""


class InvalidParameterError(EconVizError):
    """Raised when a model or canvas receives a parameter outside its
    valid domain (e.g. negative elasticity, rho=1 in CES)."""


class ExportError(EconVizError):
    """Raised when figure export fails (unsupported format, I/O error,
    or TikZ conversion issue)."""


class ParseError(EconVizError):
    """Raised when a LaTeX math string cannot be parsed into a valid
    utility function (e.g. unrecognised syntax, missing variables)."""

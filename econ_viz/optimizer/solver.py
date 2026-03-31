"""
Consumer utility maximisation subject to a linear budget constraint.

The solver handles three cases transparently:

1. **Interior tangency** — smooth preferences with a unique interior optimum
   found via constrained numerical optimisation (SLSQP).
2. **Kink solution** — for Leontief-type preferences the optimum is
   computed analytically at the intersection of the kink locus and the
   budget line.
3. **Corner solution** — for linear preferences the solver evaluates
   utility at both axis intercepts and picks the dominant corner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..enums import UtilityType
from ..exceptions import OptimizationError, InvalidParameterError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Equilibrium:
    """Result of a consumer optimisation problem.

    Attributes
    ----------
    x : float
        Optimal quantity of good *x*.
    y : float
        Optimal quantity of good *y*.
    utility : float
        Utility level attained at the optimum.
    bundle_type : str
        One of ``"interior"``, ``"kink"``, or ``"corner"``.
    """

    x: float
    y: float
    utility: float
    bundle_type: str


def solve(func, px: float, py: float, income: float) -> Equilibrium:
    """Find the utility-maximising bundle on a linear budget constraint.

    Parameters
    ----------
    func : UtilityFunction
        A utility model conforming to the protocol.
    px : float
        Price of good *x*. Must be positive.
    py : float
        Price of good *y*. Must be positive.
    income : float
        Total budget. Must be positive.

    Returns
    -------
    Equilibrium

    Raises
    ------
    InvalidParameterError
        If prices or income are non-positive.
    OptimizationError
        If the numerical solver fails to converge.
    """
    if px <= 0 or py <= 0 or income <= 0:
        raise InvalidParameterError(
            f"Prices and income must be positive (px={px}, py={py}, income={income})."
        )

    utype = getattr(func, "utility_type", UtilityType.SMOOTH)

    if utype is UtilityType.KINKED:
        return _solve_kinked(func, px, py, income)
    if utype is UtilityType.LINEAR:
        return _solve_corner(func, px, py, income)
    return _solve_interior(func, px, py, income)


# ------------------------------------------------------------------
# Strategy implementations
# ------------------------------------------------------------------

def _solve_interior(func, px: float, py: float, income: float) -> Equilibrium:
    """Smooth preferences — constrained optimisation via SLSQP.

    If the model exposes a ``lower_bounds()`` method (e.g. Stone-Geary), the
    feasible region is shifted accordingly and income is validated against the
    subsistence expenditure.
    """
    x_floor, y_floor = getattr(func, "lower_bounds", lambda: (0.0, 0.0))()

    subsistence_cost = px * x_floor + py * y_floor
    if subsistence_cost >= income:
        raise InvalidParameterError(
            f"Income ({income}) is insufficient to cover subsistence expenditure "
            f"({subsistence_cost:.4g} = px*bar_x + py*bar_y)."
        )

    x_max = income / px
    y_max = income / py
    x0 = np.array([
        x_floor + (x_max - x_floor) / 2,
        y_floor + (y_max - y_floor) / 2,
    ])

    result = minimize(
        fun=lambda v: -float(func(v[0], v[1])),
        x0=x0,
        method="SLSQP",
        bounds=[(x_floor + 1e-12, x_max), (y_floor + 1e-12, y_max)],
        constraints={"type": "eq", "fun": lambda v: px * v[0] + py * v[1] - income},
    )

    if not result.success:
        raise OptimizationError(f"SLSQP failed: {result.message}")

    xr, yr = float(result.x[0]), float(result.x[1])
    u = float(func(xr, yr))
    tol = 1e-7
    at_x_floor = abs(xr - x_floor) <= tol
    at_y_floor = abs(yr - y_floor) <= tol
    bundle_type = "boundary" if at_x_floor or at_y_floor else "interior"
    logger.debug("%s solution: x=%.6f, y=%.6f, U=%.6f", bundle_type.title(), xr, yr, u)
    return Equilibrium(x=xr, y=yr, utility=u, bundle_type=bundle_type)


def _solve_kinked(func, px: float, py: float, income: float) -> Equilibrium:
    """Kinked preferences — analytic solution at kink-locus / budget intersection.

    For Leontief U = min(ax, by), the optimum is where ax = by AND
    px*x + py*y = I. Solving: x = I / (px + py * a/b), y = (a/b) * x.
    """
    slopes = func.ray_slopes()
    if not slopes:
        return _solve_interior(func, px, py, income)

    slope = slopes[0]  # dy/dx along kink locus
    xr = income / (px + py * slope)
    yr = slope * xr
    u = float(func(xr, yr))
    logger.debug("Kink solution: x=%.6f, y=%.6f, U=%.6f", xr, yr, u)
    return Equilibrium(x=xr, y=yr, utility=u, bundle_type="kink")


def _solve_corner(func, px: float, py: float, income: float) -> Equilibrium:
    """Linear preferences — compare utility at the two axis intercepts."""
    x_corner = income / px
    y_corner = income / py

    u_x = float(func(x_corner, 0))
    u_y = float(func(0, y_corner))

    if u_x >= u_y:
        logger.debug("Corner solution at x-axis: x=%.6f, U=%.6f", x_corner, u_x)
        return Equilibrium(x=x_corner, y=0.0, utility=u_x, bundle_type="corner")
    logger.debug("Corner solution at y-axis: y=%.6f, U=%.6f", y_corner, u_y)
    return Equilibrium(x=0.0, y=y_corner, utility=u_y, bundle_type="corner")

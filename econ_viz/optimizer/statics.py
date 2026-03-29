"""
Comparative statics for consumer demand.

:func:`comparative_statics` numerically computes the six partial derivatives
of the Marshallian demands with respect to prices and income by applying
central finite differences to :func:`~econ_viz.optimizer.solver.solve`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .solver import solve
from ..exceptions import InvalidParameterError
from ..logging import get_logger

logger = get_logger(__name__)

_DEFAULT_H = 1e-3


@dataclass(frozen=True)
class ComparativeStatics:
    """Partial derivatives of Marshallian demands.

    Attributes
    ----------
    dx_dpx : float  ∂x*/∂p_x
    dx_dpy : float  ∂x*/∂p_y
    dx_dI  : float  ∂x*/∂I  (Engel slope for x)
    dy_dpx : float  ∂y*/∂p_x
    dy_dpy : float  ∂y*/∂p_y
    dy_dI  : float  ∂y*/∂I  (Engel slope for y)
    """

    dx_dpx: float
    dx_dpy: float
    dx_dI: float
    dy_dpx: float
    dy_dpy: float
    dy_dI: float

    def __repr__(self) -> str:  # pragma: no cover
        rows = [
            ("∂x*/∂pₓ", self.dx_dpx),
            ("∂x*/∂pᵧ", self.dx_dpy),
            ("∂x*/∂I ", self.dx_dI),
            ("∂y*/∂pₓ", self.dy_dpx),
            ("∂y*/∂pᵧ", self.dy_dpy),
            ("∂y*/∂I ", self.dy_dI),
        ]
        lines = ["ComparativeStatics"]
        lines.append("─" * 26)
        for label, val in rows:
            lines.append(f"  {label}  {val:+.6f}")
        lines.append("─" * 26)
        return "\n".join(lines)


def comparative_statics(
    func,
    px: float,
    py: float,
    income: float,
    h: float = _DEFAULT_H,
) -> ComparativeStatics:
    """Compute comparative statics for a consumer optimisation problem.

    Uses central finite differences to estimate all six partial derivatives
    of the Marshallian demands with respect to prices and income.

    Step sizes are *relative*: for each parameter ``p`` the perturbation is
    ``max(h * p, h)`` so that the step is never degenerate when the parameter
    value is close to zero.

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
    h : float
        Relative step size for finite differences. The actual perturbation for
        parameter *p* is ``max(h * p, h)``, so it scales with the parameter
        value.  Default ``1e-3`` — large enough to dominate SLSQP variable
        noise while keeping the central-difference truncation error O(h²).

    Returns
    -------
    ComparativeStatics

    Raises
    ------
    InvalidParameterError
        If prices or income are non-positive.

    Warns
    -----
    UserWarning
        If sign consistency checks fail (e.g. ∂x*/∂p_x > 0, ∂x*/∂I < 0).
    """
    if px <= 0 or py <= 0 or income <= 0:
        raise InvalidParameterError(
            f"Prices and income must be positive (px={px}, py={py}, income={income})."
        )

    def _deriv(param: str) -> tuple[float, float]:
        """Return (dx/dparam, dy/dparam) via central differences."""
        base = {"px": px, "py": py, "income": income}
        val = base[param]
        step = max(h * val, h)

        lo = {**base, param: val - step}
        hi = {**base, param: val + step}

        eq_lo = solve(func, **lo)
        eq_hi = solve(func, **hi)

        dx = (eq_hi.x - eq_lo.x) / (2 * step)
        dy = (eq_hi.y - eq_lo.y) / (2 * step)
        return dx, dy

    dx_dpx, dy_dpx = _deriv("px")
    dx_dpy, dy_dpy = _deriv("py")
    dx_dI,  dy_dI  = _deriv("income")

    logger.debug(
        "ComparativeStatics at (px=%.4g, py=%.4g, I=%.4g): "
        "dx/dpx=%.4g  dx/dpy=%.4g  dx/dI=%.4g  "
        "dy/dpx=%.4g  dy/dpy=%.4g  dy/dI=%.4g",
        px, py, income,
        dx_dpx, dx_dpy, dx_dI,
        dy_dpx, dy_dpy, dy_dI,
    )

    cs = ComparativeStatics(
        dx_dpx=dx_dpx,
        dx_dpy=dx_dpy,
        dx_dI=dx_dI,
        dy_dpx=dy_dpx,
        dy_dpy=dy_dpy,
        dy_dI=dy_dI,
    )

    _warn_sign_violations(cs)
    return cs


def _warn_sign_violations(cs: ComparativeStatics) -> None:
    """Emit UserWarnings for economically unusual sign patterns."""
    _tol = 1e-8

    if cs.dx_dpx > _tol:
        warnings.warn(
            f"∂x*/∂pₓ = {cs.dx_dpx:.4g} > 0 — Giffen good or numerical artefact.",
            UserWarning,
            stacklevel=3,
        )
    if cs.dy_dpy > _tol:
        warnings.warn(
            f"∂y*/∂pᵧ = {cs.dy_dpy:.4g} > 0 — Giffen good or numerical artefact.",
            UserWarning,
            stacklevel=3,
        )
    if cs.dx_dI < -_tol:
        warnings.warn(
            f"∂x*/∂I = {cs.dx_dI:.4g} < 0 — inferior good for x.",
            UserWarning,
            stacklevel=3,
        )
    if cs.dy_dI < -_tol:
        warnings.warn(
            f"∂y*/∂I = {cs.dy_dI:.4g} < 0 — inferior good for y.",
            UserWarning,
            stacklevel=3,
        )

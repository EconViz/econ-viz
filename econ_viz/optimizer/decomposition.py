"""Price-effect decomposition helpers (Hicks and Slutsky compensation)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize

from ..exceptions import InvalidParameterError, OptimizationError
from .slutsky import SlutskyMatrix, slutsky_matrix
from .solver import Equilibrium, solve

_EPS = 1e-12
_BOUNDARY_TOL = 1e-7


class DecompositionMethod(str, Enum):
    """Compensation rule used to construct the intermediate bundle B."""

    HICKS = "hicks"
    SLUTSKY = "slutsky"

    @classmethod
    def coerce(cls, method: DecompositionMethod | str) -> DecompositionMethod:
        """Normalise string or enum input into a method enum."""
        if isinstance(method, cls):
            return method
        if isinstance(method, str):
            token = method.strip().lower()
            if token in {"hicks", "h"}:
                return cls.HICKS
            if token in {"slutsky", "s"}:
                return cls.SLUTSKY
        raise InvalidParameterError(
            "method must be DecompositionMethod.HICKS or DecompositionMethod.SLUTSKY."
        )


@dataclass(frozen=True)
class PriceEffectDecomposition:
    """Structured result for a price-effect decomposition."""

    method: DecompositionMethod
    px_before: float
    px_after: float
    py: float
    income: float
    compensated_income: float
    A: Equilibrium
    B: Equilibrium
    C: Equilibrium
    substitution_effect: tuple[float, float]
    income_effect: tuple[float, float]
    total_effect: tuple[float, float]
    slutsky_matrix: SlutskyMatrix

    def vector_identity_holds(self, tol: float = 1e-9) -> bool:
        """Return ``True`` when ``(A->B) + (B->C) == (A->C)`` within tolerance."""
        lhs = np.asarray(self.substitution_effect) + np.asarray(self.income_effect)
        rhs = np.asarray(self.total_effect)
        return bool(np.allclose(lhs, rhs, atol=tol, rtol=0.0))


def decompose_price_effect(
    func,
    *,
    px: tuple[float, float],
    py: float,
    income: float,
    method: DecompositionMethod | str = DecompositionMethod.SLUTSKY,
) -> PriceEffectDecomposition:
    """Decompose a price change into substitution and income effects.

    Parameters
    ----------
    func : UtilityFunction
        Utility model conforming to the project protocol.
    px : tuple[float, float]
        ``(px_before, px_after)`` price pair.
    py : float
        Price of good ``y``.
    income : float
        Nominal income.
    method : DecompositionMethod or str
        Compensation rule for intermediate bundle ``B``.
    """
    px_before, px_after = _validate_px_pair(px)
    if py <= 0 or income <= 0:
        raise InvalidParameterError(
            f"Prices and income must be positive (py={py}, income={income})."
        )

    method_enum = DecompositionMethod.coerce(method)

    bundle_A = solve(func, px=px_before, py=py, income=income)
    bundle_C = solve(func, px=px_after, py=py, income=income)

    if method_enum is DecompositionMethod.SLUTSKY:
        compensated_income = px_after * bundle_A.x + py * bundle_A.y
        bundle_B = solve(func, px=px_after, py=py, income=compensated_income)
    else:
        bundle_B, compensated_income = _solve_hicks_compensated_bundle(
            func,
            px_after=px_after,
            py=py,
            reference=bundle_A,
        )

    substitution_effect = (bundle_B.x - bundle_A.x, bundle_B.y - bundle_A.y)
    income_effect = (bundle_C.x - bundle_B.x, bundle_C.y - bundle_B.y)
    total_effect = (bundle_C.x - bundle_A.x, bundle_C.y - bundle_A.y)

    decomposition = PriceEffectDecomposition(
        method=method_enum,
        px_before=px_before,
        px_after=px_after,
        py=py,
        income=income,
        compensated_income=float(compensated_income),
        A=bundle_A,
        B=bundle_B,
        C=bundle_C,
        substitution_effect=substitution_effect,
        income_effect=income_effect,
        total_effect=total_effect,
        slutsky_matrix=slutsky_matrix(func, px=px_before, py=py, income=income),
    )

    if not decomposition.vector_identity_holds():
        raise OptimizationError(
            "Price decomposition identity failed: (A->B)+(B->C) != (A->C)."
        )
    return decomposition


def _validate_px_pair(px: tuple[float, float]) -> tuple[float, float]:
    try:
        px_before_raw, px_after_raw = px
        px_before, px_after = float(px_before_raw), float(px_after_raw)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise InvalidParameterError(
            "px must be a tuple (px_before, px_after) with two positive values."
        ) from exc
    if px_before <= 0 or px_after <= 0:
        raise InvalidParameterError(
            f"Both px values must be positive (px_before={px_before}, px_after={px_after})."
        )
    return px_before, px_after


def _solve_hicks_compensated_bundle(
    func,
    *,
    px_after: float,
    py: float,
    reference: Equilibrium,
) -> tuple[Equilibrium, float]:
    """Return Hicks compensated bundle B and its compensation income."""
    x_floor, y_floor = getattr(func, "lower_bounds", lambda: (0.0, 0.0))()
    x0 = np.array(
        [max(reference.x, x_floor + _EPS), max(reference.y, y_floor + _EPS)],
        dtype=float,
    )
    u_target = float(reference.utility)

    result = minimize(
        fun=lambda v: px_after * float(v[0]) + py * float(v[1]),
        x0=x0,
        method="SLSQP",
        bounds=[(x_floor + _EPS, None), (y_floor + _EPS, None)],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda v: float(func(float(v[0]), float(v[1])) - u_target),
            },
        ],
    )
    if not result.success:
        raise OptimizationError(f"Hicks compensation SLSQP failed: {result.message}")

    x_b, y_b = float(result.x[0]), float(result.x[1])
    utility_b = float(func(x_b, y_b))
    if utility_b + 1e-6 < u_target:
        raise OptimizationError(
            "Hicks compensation failed: compensated bundle utility below target utility."
        )

    is_boundary = (
        abs(x_b - x_floor) <= _BOUNDARY_TOL
        or abs(y_b - y_floor) <= _BOUNDARY_TOL
    )
    bundle_type = "boundary" if is_boundary else "interior"
    compensated_income = px_after * x_b + py * y_b
    return (
        Equilibrium(x=x_b, y=y_b, utility=utility_b, bundle_type=bundle_type),
        float(compensated_income),
    )

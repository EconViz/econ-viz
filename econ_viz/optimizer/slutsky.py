"""Slutsky decomposition helpers built on top of Marshallian demand responses."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .comparative import ComparativeStatics, comparative_statics
from .solver import solve

_DEFAULT_H = 1e-3
_DEFAULT_TOL = 5e-2


@dataclass(frozen=True)
class SlutskyMatrix:
    """Two-good Slutsky substitution matrix.

    Attributes
    ----------
    s_xx : float
        ``∂x^h/∂p_x = ∂x*/∂p_x + x* ∂x*/∂I``
    s_xy : float
        ``∂x^h/∂p_y = ∂x*/∂p_y + y* ∂x*/∂I``
    s_yx : float
        ``∂y^h/∂p_x = ∂y*/∂p_x + x* ∂y*/∂I``
    s_yy : float
        ``∂y^h/∂p_y = ∂y*/∂p_y + y* ∂y*/∂I``
    """

    s_xx: float
    s_xy: float
    s_yx: float
    s_yy: float

    def as_array(self) -> np.ndarray:
        """Return the matrix as a 2x2 NumPy array."""
        return np.array(
            [[self.s_xx, self.s_xy], [self.s_yx, self.s_yy]],
            dtype=float,
        )

    def is_symmetric(self, tol: float = _DEFAULT_TOL) -> bool:
        """Return ``True`` when ``S_xy`` and ``S_yx`` agree within tolerance."""
        return bool(np.isclose(self.s_xy, self.s_yx, atol=tol, rtol=0.0))

    def is_negative_semidefinite(self, tol: float = _DEFAULT_TOL) -> bool:
        """Return ``True`` when all eigenvalues are non-positive within tolerance."""
        eigenvalues = np.linalg.eigvalsh(self.as_array())
        return bool(np.all(eigenvalues <= tol))

    def satisfies_homogeneity(
        self,
        px: float,
        py: float,
        tol: float = _DEFAULT_TOL,
    ) -> bool:
        """Return ``True`` when ``S @ p ≈ 0`` within tolerance."""
        residual = self.as_array() @ np.array([px, py], dtype=float)
        return bool(np.all(np.abs(residual) <= tol))

    def validation_failures(
        self,
        *,
        px: float,
        py: float,
        tol: float = _DEFAULT_TOL,
    ) -> list[str]:
        """Return the names of any theoretical checks that fail."""
        failures: list[str] = []
        if not self.is_symmetric(tol=tol):
            failures.append("symmetry")
        if not self.is_negative_semidefinite(tol=tol):
            failures.append("negative semidefinite")
        if not self.satisfies_homogeneity(px=px, py=py, tol=tol):
            failures.append("homogeneity")
        return failures

    def __repr__(self) -> str:  # pragma: no cover
        lines = ["SlutskyMatrix", "─" * 28]
        lines.append(f"  [ {self.s_xx:+.6f}  {self.s_xy:+.6f} ]")
        lines.append(f"  [ {self.s_yx:+.6f}  {self.s_yy:+.6f} ]")
        lines.append("─" * 28)
        return "\n".join(lines)


def slutsky_matrix(
    func,
    px: float,
    py: float,
    income: float,
    h: float = _DEFAULT_H,
) -> SlutskyMatrix:
    """Compute the 2x2 Slutsky substitution matrix.

    Uses the Slutsky equation

    ``S = D_p x + D_I x · x^T``

    where ``D_p x`` is the matrix of Marshallian price derivatives,
    ``D_I x`` is the vector of income derivatives, and ``x`` is the
    Marshallian demand vector at ``(px, py, income)``.
    """
    cs: ComparativeStatics = comparative_statics(func, px=px, py=py, income=income, h=h)
    eq = solve(func, px=px, py=py, income=income)

    matrix = SlutskyMatrix(
        s_xx=cs.dx_dpx + eq.x * cs.dx_dI,
        s_xy=cs.dx_dpy + eq.y * cs.dx_dI,
        s_yx=cs.dy_dpx + eq.x * cs.dy_dI,
        s_yy=cs.dy_dpy + eq.y * cs.dy_dI,
    )
    failures = matrix.validation_failures(px=px, py=py)
    if failures:
        warnings.warn(
            "Slutsky matrix theoretical checks failed: " + ", ".join(failures),
            UserWarning,
            stacklevel=2,
        )
    return matrix

"""
Homogeneity analysis for utility and production functions.

Provides :class:`HomogeneityAnalyzer`, which numerically tests whether a
callable ``f(x, y)`` satisfies homogeneity of degree *k*, verifies Euler's
theorem, checks homotheticity, and confirms that Marshallian demands are
homogeneous of degree 0 in prices and income.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..enums import ReturnsToScale


@dataclass(frozen=True)
class HomogeneityResult:
    """Summary of a homogeneity analysis.

    Attributes
    ----------
    degree : float or None
        Estimated homogeneity degree *k*, or ``None`` if no consistent
        degree was found across sampled bundles.
    is_homogeneous : bool
        ``True`` when *degree* is not ``None``.
    returns_to_scale : ReturnsToScale
        Scaling classification derived from *degree*.
    """

    degree: float | None
    is_homogeneous: bool
    returns_to_scale: ReturnsToScale


class HomogeneityAnalyzer:
    """Numerical homogeneity analyser for a two-good utility function.

    Parameters
    ----------
    func : callable
        A utility model ``f(x, y)`` accepting scalar floats and returning
        a scalar float.  Any object satisfying the ``UtilityFunction``
        protocol is accepted.
    tol : float
        Tolerance for degree consistency and Euler residual checks.
    n_samples : int
        Number of base bundles sampled when estimating the degree.
    """

    def __init__(self, func, *, tol: float = 1e-4, n_samples: int = 8) -> None:
        self._func = func
        self._tol = tol
        self._n_samples = n_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def degree(self) -> HomogeneityResult:
        """Estimate the homogeneity degree *k*.

        Samples *n_samples* base bundles $(x_0, y_0)$ drawn from a
        log-uniform grid on $[0.5, 5]^2$ and three scaling factors
        $\\lambda \\in \\{0.5, 2.0, 4.0\\}$.  For each combination:

        .. math::

            k = \\frac{\\ln f(\\lambda x_0,\\, \\lambda y_0) - \\ln f(x_0, y_0)}{\\ln \\lambda}

        Returns ``None`` in the result when the estimated *k* values are
        inconsistent (standard deviation > *tol*).

        Returns
        -------
        HomogeneityResult
        """
        rng = np.random.default_rng(0)
        xs = np.exp(rng.uniform(np.log(0.5), np.log(5.0), self._n_samples))
        ys = np.exp(rng.uniform(np.log(0.5), np.log(5.0), self._n_samples))
        lambdas = [0.5, 2.0, 4.0]

        estimates: list[float] = []
        for x0, y0 in zip(xs, ys):
            f0 = float(self._func(x0, y0))
            if f0 <= 0:
                continue
            for lam in lambdas:
                fl = float(self._func(lam * x0, lam * y0))
                if fl <= 0:
                    continue
                estimates.append((np.log(fl) - np.log(f0)) / np.log(lam))

        if not estimates:
            return HomogeneityResult(
                degree=None,
                is_homogeneous=False,
                returns_to_scale=ReturnsToScale.NOT_HOMOGENEOUS,
            )

        arr = np.array(estimates)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3 or arr.std() > self._tol:
            return HomogeneityResult(
                degree=None,
                is_homogeneous=False,
                returns_to_scale=ReturnsToScale.NOT_HOMOGENEOUS,
            )

        k = float(arr.mean())
        return HomogeneityResult(
            degree=k,
            is_homogeneous=True,
            returns_to_scale=ReturnsToScale.from_degree(k, tol=self._tol),
        )

    def euler_check(self, x: float, y: float) -> float:
        """Return the relative Euler residual at bundle *(x, y)*.

        For a degree-*k* homogeneous function the residual should be
        near zero:

        .. math::

            \\varepsilon = \\frac{\\left| x \\partial_x f + y \\partial_y f
                           - k \\cdot f(x, y) \\right|}{|f(x, y)|}

        Parameters
        ----------
        x, y : float
            Bundle coordinates (must be strictly positive).

        Returns
        -------
        float
            Relative residual.  Values below *tol* indicate the theorem
            holds numerically.
        """
        result = self.degree()
        if not result.is_homogeneous or result.degree is None:
            return float("nan")

        k = result.degree
        h = 1e-5
        df_dx = (float(self._func(x + h, y)) - float(self._func(x - h, y))) / (2 * h)
        df_dy = (float(self._func(x, y + h)) - float(self._func(x, y - h))) / (2 * h)
        fxy = float(self._func(x, y))

        lhs = x * df_dx + y * df_dy
        return abs(lhs - k * fxy) / abs(fxy)

    def is_homothetic(self, *, n_samples: int = 6) -> bool:
        """Test whether the function is homothetic.

        A function is homothetic if the MRS depends only on the ratio
        $x / y$, not on the scale of the bundle:

        .. math::

            \\text{MRS}(x_0, y_0) \\approx \\text{MRS}(\\lambda x_0,\\, \\lambda y_0)

        Samples *n_samples* base bundles and tests with
        $\\lambda \\in \\{0.5, 2.0, 4.0\\}$.

        Returns
        -------
        bool
        """
        rng = np.random.default_rng(1)
        xs = np.exp(rng.uniform(np.log(0.5), np.log(4.0), n_samples))
        ys = np.exp(rng.uniform(np.log(0.5), np.log(4.0), n_samples))
        lambdas = [0.5, 2.0, 4.0]
        h = 1e-5

        def _mrs(x, y) -> float:
            df_dx = (float(self._func(x + h, y)) - float(self._func(x - h, y))) / (2 * h)
            df_dy = (float(self._func(x, y + h)) - float(self._func(x, y - h))) / (2 * h)
            if abs(df_dy) < 1e-12:
                return float("nan")
            return df_dx / df_dy

        for x0, y0 in zip(xs, ys):
            mrs0 = _mrs(x0, y0)
            if np.isnan(mrs0):
                continue
            for lam in lambdas:
                mrs_scaled = _mrs(lam * x0, lam * y0)
                if np.isnan(mrs_scaled):
                    continue
                if abs(mrs_scaled - mrs0) > self._tol * (1 + abs(mrs0)):
                    return False
        return True

    def demand_degree_zero(
        self,
        px: float,
        py: float,
        income: float,
        *,
        scales: tuple[float, ...] = (0.5, 2.0, 5.0),
    ) -> bool:
        """Verify Marshallian demands are homogeneous of degree 0 in $(p_x, p_y, I)$.

        Checks that scaling all prices and income by $t$ leaves the
        optimal bundle unchanged:

        .. math::

            x^{*}(t p_x,\\, t p_y,\\, t I) = x^{*}(p_x, p_y, I)

        Parameters
        ----------
        px, py : float
            Reference prices (must be positive).
        income : float
            Reference income (must be positive).
        scales : tuple of float
            Scale factors $t$ to test.

        Returns
        -------
        bool
        """
        from ..optimizer import solve

        eq0 = solve(self._func, px=px, py=py, income=income)
        for t in scales:
            eq_t = solve(self._func, px=t * px, py=t * py, income=t * income)
            if abs(eq_t.x - eq0.x) > self._tol * (1 + abs(eq0.x)):
                return False
            if abs(eq_t.y - eq0.y) > self._tol * (1 + abs(eq0.y)):
                return False
        return True

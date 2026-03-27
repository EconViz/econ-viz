"""
Concrete utility function implementations.

All classes conform to the :class:`~econ_viz.models.protocol.UtilityFunction`
protocol:

* **Callable** — ``model(x, y)`` returns element-wise utility values,
  accepting both scalars and :class:`numpy.ndarray` inputs.
* **utility_type** — a :class:`~econ_viz.enums.UtilityType` member describing
  the qualitative shape of the indifference map.
* **ray_slopes()** — slopes for economically meaningful rays from the origin.
* **kink_points(levels)** — coordinates of non-differentiable kink points on
  specified contour levels (non-empty only for ``KINKED`` preferences).
"""

import numpy as np
from dataclasses import dataclass

from ..enums import UtilityType
from ..exceptions import InvalidParameterError


@dataclass
class CobbDouglas:
    """Cobb-Douglas utility: U(x, y) = x^alpha * y^beta.

    The expansion path under equal prices has slope beta/alpha, representing
    the optimal consumption ratio when the consumer faces a unit price ratio.

    Parameters
    ----------
    alpha : float
        Output elasticity of good *x*.
    beta : float
        Output elasticity of good *y*.
    """

    alpha: float = 0.5
    beta: float = 0.5

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        return (x ** self.alpha) * (y ** self.beta)

    def ray_slopes(self) -> list[float]:
        """Return the expansion-path slope beta/alpha (equal-price case)."""
        return [self.beta / self.alpha]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


@dataclass
class Leontief:
    """Leontief (perfect complements) utility: U(x, y) = min(ax, by).

    Indifference curves exhibit right-angle kinks along the ray y = (a/b)x.
    At utility level U, the kink occurs at (U/a, U/b).

    Parameters
    ----------
    a : float
        Coefficient on good *x*.
    b : float
        Coefficient on good *y*.
    """

    a: float = 1.0
    b: float = 1.0

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.KINKED

    def __call__(self, x, y):
        return np.minimum(self.a * x, self.b * y)

    def ray_slopes(self) -> list[float]:
        """Return the kink-locus slope a/b."""
        return [self.a / self.b]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        """Return kink vertices (U/a, U/b) for each contour level."""
        return [(u / self.a, u / self.b) for u in levels]


@dataclass
class PerfectSubstitutes:
    """Perfect substitutes utility: U(x, y) = ax + by.

    Indifference curves are linear with slope -a/b. The ray at slope a/b
    from the origin represents the marginal rate of substitution direction.

    Parameters
    ----------
    a : float
        Marginal utility of good *x*.
    b : float
        Marginal utility of good *y*.
    """

    a: float = 1.0
    b: float = 1.0

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.LINEAR

    def __call__(self, x, y):
        return self.a * x + self.b * y

    def ray_slopes(self) -> list[float]:
        """Return the MRS-derived slope a/b."""
        return [self.a / self.b]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


@dataclass
class CES:
    """Constant Elasticity of Substitution utility:

    U(x, y) = (alpha * x^rho + beta * y^rho)^{1/rho}

    The elasticity of substitution is sigma = 1/(1 - rho). Special cases:

    * rho -> 0  : Cobb-Douglas
    * rho -> -inf : Leontief
    * rho = 1    : perfect substitutes

    The equal-price expansion path has slope (alpha/beta)^{1/(1-rho)}.

    Parameters
    ----------
    alpha : float
        Share parameter for good *x*.
    beta : float
        Share parameter for good *y*.
    rho : float
        Substitution parameter. Must not equal 1 (use
        :class:`PerfectSubstitutes` instead).
    """

    alpha: float = 0.5
    beta: float = 0.5
    rho: float = 0.5

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        return (self.alpha * x ** self.rho + self.beta * y ** self.rho) ** (1 / self.rho)

    def ray_slopes(self) -> list[float]:
        """Return the expansion-path slope (alpha/beta)^{1/(1-rho)}.

        Falls back to beta/alpha (Cobb-Douglas limit) when rho is near zero,
        and raises ValueError when rho equals 1.
        """
        if abs(self.rho - 1.0) < 1e-9:
            raise InvalidParameterError(
                "CES with rho=1 is equivalent to perfect substitutes; "
                "use PerfectSubstitutes instead."
            )
        if abs(self.rho) < 1e-9:
            # Cobb-Douglas limit: slope = beta / alpha
            return [self.beta / self.alpha]
        return [(self.alpha / self.beta) ** (1 / (1 - self.rho))]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []

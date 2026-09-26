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

Class order
-----------
1. CobbDouglas      — workhorse smooth preferences
2. CES              — generalises Cobb-Douglas via substitution parameter
3. PerfectSubstitutes — linear (rho=1 limit of CES)
4. Leontief         — kinked (rho=-inf limit of CES)
5. Translog         — flexible second-order log approximation
6. QuasiLinear      — one linear good, no income effect
7. StoneGeary       — subsistence extension of Cobb-Douglas
8. Satiation        — bliss-point / non-monotone preferences
9. Haagsma          — inferior good x, Giffen at high enough income
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..enums import UtilityType
from ..exceptions import InvalidParameterError


def _require_positive(model: str, **parameters: float) -> None:
    """Validate finite, strictly positive model parameters."""
    invalid = [name for name, value in parameters.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        names = ", ".join(invalid)
        raise InvalidParameterError(f"{model}: {names} must be finite and positive.")


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

    def __post_init__(self) -> None:
        _require_positive("CobbDouglas", alpha=self.alpha, beta=self.beta)

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        return (x**self.alpha) * (y**self.beta)

    def ray_slopes(self) -> list[float]:
        """Return the expansion-path slope beta/alpha (equal-price case)."""
        return [self.beta / self.alpha]

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

    The equal-price expansion path has slope (beta/alpha)^{1/(1-rho)}.

    Parameters
    ----------
    alpha : float
        Share parameter for good *x*.
    beta : float
        Share parameter for good *y*.
    rho : float
        Substitution parameter below 1. At ``rho=0``, ``alpha`` and ``beta``
        must sum to 1 so the Cobb-Douglas limit is well-defined.
    """

    alpha: float = 0.5
    beta: float = 0.5
    rho: float = 0.5

    def __post_init__(self) -> None:
        _require_positive("CES", alpha=self.alpha, beta=self.beta)
        if not np.isfinite(self.rho) or self.rho >= 1.0:
            raise InvalidParameterError("CES: rho must be finite and less than 1; use PerfectSubstitutes for rho=1.")
        if abs(self.rho) < 1e-9 and not np.isclose(self.alpha + self.beta, 1.0):
            raise InvalidParameterError("CES: alpha and beta must sum to 1 when rho=0.")

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        if abs(self.rho) < 1e-9:
            return x**self.alpha * y**self.beta
        return (self.alpha * x**self.rho + self.beta * y**self.rho) ** (1 / self.rho)

    def ray_slopes(self) -> list[float]:
        """Return the expansion-path slope (beta/alpha)^{1/(1-rho)}.

        Falls back to beta/alpha (Cobb-Douglas limit) when rho is near zero.
        """
        if abs(self.rho) < 1e-9:
            # Cobb-Douglas limit: slope = beta / alpha
            return [self.beta / self.alpha]
        return [(self.beta / self.alpha) ** (1 / (1 - self.rho))]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


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

    def __post_init__(self) -> None:
        _require_positive("PerfectSubstitutes", a=self.a, b=self.b)

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

    def __post_init__(self) -> None:
        _require_positive("Leontief", a=self.a, b=self.b)

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
class Translog:
    """Transcendental logarithmic utility function.

    The *translog* specification is a flexible second-order approximation to
    any smooth utility function in log-space:

    .. math::

        \\ln U(x, y) = \\alpha_0
                     + \\alpha_x \\ln x + \\alpha_y \\ln y
                     + \\tfrac{1}{2} \\beta_{xx} (\\ln x)^2
                     + \\tfrac{1}{2} \\beta_{yy} (\\ln y)^2
                     + \\beta_{xy} \\ln x \\cdot \\ln y

    Setting ``beta_xx = beta_yy = beta_xy = 0`` recovers Cobb-Douglas.

    Parameters
    ----------
    alpha_x : float
        First-order coefficient on :math:`\\ln x`.
    alpha_y : float
        First-order coefficient on :math:`\\ln y`.
    beta_xx : float
        Second-order own-effect on :math:`\\ln x`.  Default 0.
    beta_yy : float
        Second-order own-effect on :math:`\\ln y`.  Default 0.
    beta_xy : float
        Cross-effect :math:`\\ln x \\cdot \\ln y`.  Default 0.
    alpha_0 : float
        Intercept in log-utility.  Scales the overall utility level.
    """

    alpha_x: float = 0.5
    alpha_y: float = 0.5
    beta_xx: float = 0.0
    beta_yy: float = 0.0
    beta_xy: float = 0.0
    alpha_0: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha_x <= 0 or self.alpha_y <= 0:
            raise InvalidParameterError("Translog: alpha_x and alpha_y must be positive.")

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        lx = np.log(x)
        ly = np.log(y)
        ln_u = (
            self.alpha_0
            + self.alpha_x * lx
            + self.alpha_y * ly
            + 0.5 * self.beta_xx * lx**2
            + 0.5 * self.beta_yy * ly**2
            + self.beta_xy * lx * ly
        )
        return np.exp(ln_u)

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


def _validate_v_func(v_func: Callable, name: str = "v_func") -> None:
    """Check f'(z) > 0 and f''(z) < 0 on z in [0.1, 10] via central differences."""
    h = 1e-5
    z = np.linspace(0.1, 10.0, 100)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_minus = np.asarray(v_func(z - h), dtype=float)
        f_center = np.asarray(v_func(z), dtype=float)
        f_plus = np.asarray(v_func(z + h), dtype=float)

    fprime = (f_plus - f_minus) / (2 * h)
    fdouble = (f_plus - 2 * f_center + f_minus) / (h**2)

    if np.any(fprime <= -1e-8):
        raise ValueError(f"{name}: monotonicity violated -- f'(z) <= 0 detected.")
    if np.any(fdouble >= 1e-8):
        raise ValueError(f"{name}: diminishing marginal utility violated -- f''(z) >= 0 detected.")


@dataclass
class QuasiLinear:
    """Quasi-linear utility: U(x, y) = f(x) + y  or  U(x, y) = x + f(y).

    One good enters linearly; the other enters through a strictly concave,
    strictly increasing transformation f.  This specification produces
    indifference curves with the same shape at every income level (no income
    effect on the non-linear good).

    Parameters
    ----------
    v_func : Callable
        Strictly increasing, strictly concave scalar function f(z).
        Validated numerically on z in [0.1, 10] via central differences.
        Defaults to ``numpy.log``.
    linear_in : {'x', 'y'}
        Which good enters *linearly*.  Defaults to ``'y'``, giving
        U(x, y) = f(x) + y.
    """

    v_func: Callable = field(default=np.log)
    linear_in: str = "y"

    def __post_init__(self) -> None:
        if self.linear_in not in ("x", "y"):
            raise InvalidParameterError("linear_in must be 'x' or 'y'.")
        _validate_v_func(self.v_func, name="v_func")

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        if self.linear_in == "y":
            # U = f(x) + y
            return self.v_func(x) + y
        else:
            # U = x + f(y)
            return x + self.v_func(y)

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


@dataclass
class StoneGeary:
    """Stone-Geary utility: U(x, y) = (x - bar_x)^alpha * (y - bar_y)^beta.

    Utility is defined only in the supernumerary region x > bar_x, y > bar_y.
    The consumer first secures subsistence quantities (bar_x, bar_y) and then
    allocates the remaining *supernumerary* income like a Cobb-Douglas
    consumer.  Marshallian demands are:

        x* = bar_x + [alpha / (alpha + beta)] * (I - px*bar_x - py*bar_y) / px
        y* = bar_y + [beta  / (alpha + beta)] * (I - px*bar_x - py*bar_y) / py

    Parameters
    ----------
    alpha : float
        Expenditure share on supernumerary *x*.  Must be positive.
    beta : float
        Expenditure share on supernumerary *y*.  Must be positive.
    bar_x : float
        Subsistence quantity of good *x*.  Must be non-negative.
    bar_y : float
        Subsistence quantity of good *y*.  Must be non-negative.
    """

    alpha: float = 0.5
    beta: float = 0.5
    bar_x: float = 1.0
    bar_y: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise InvalidParameterError("StoneGeary: alpha and beta must be positive.")
        if self.bar_x < 0 or self.bar_y < 0:
            raise InvalidParameterError("StoneGeary: bar_x and bar_y must be non-negative.")

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        dx = np.asarray(x, dtype=float) - self.bar_x
        dy = np.asarray(y, dtype=float) - self.bar_y
        with np.errstate(invalid="ignore"):
            result = np.where(
                (dx > 0) & (dy > 0),
                dx**self.alpha * dy**self.beta,
                np.nan,
            )
        return float(result) if result.ndim == 0 else result

    def lower_bounds(self) -> tuple[float, float]:
        """Return the minimum feasible (x, y) -- the subsistence point."""
        return (self.bar_x, self.bar_y)

    def subsistence_lines(self) -> tuple[float, float]:
        """Return (bar_x, bar_y) for drawing subsistence reference lines on the canvas."""
        return (self.bar_x, self.bar_y)

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


@dataclass
class Satiation:
    """Satiation (bliss-point) utility: U(x, y) = -a(x - x*)^2 - b(y - y*)^2.

    Utility rises as the bundle approaches the bliss point (x*, y*) and falls
    away from it in all directions.  Indifference curves are closed ellipses
    centred on the bliss point.

    Parameters
    ----------
    bliss_x : float
        x-coordinate of the bliss point x*.
    bliss_y : float
        y-coordinate of the bliss point y*.
    a : float
        Curvature (penalty weight) along the x-axis.  Must be positive.
    b : float
        Curvature (penalty weight) along the y-axis.  Must be positive.
    """

    bliss_x: float = 5.0
    bliss_y: float = 5.0
    a: float = 1.0
    b: float = 1.0

    def __post_init__(self) -> None:
        _require_positive("Satiation", a=self.a, b=self.b)
        if not np.isfinite(self.bliss_x) or not np.isfinite(self.bliss_y) or self.bliss_x < 0 or self.bliss_y < 0:
            raise InvalidParameterError("Satiation: bliss_x and bliss_y must be finite and non-negative.")

    @property
    def budget_may_be_slack(self) -> bool:
        """Return whether the optimum may leave income unspent."""
        return True

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        return -self.a * (x - self.bliss_x) ** 2 - self.b * (y - self.bliss_y) ** 2

    def ray_slopes(self) -> list[float]:
        """Return the slope of the ray from origin through the bliss point."""
        if self.bliss_x == 0:
            return []
        return [self.bliss_y / self.bliss_x]

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []


@dataclass
class Haagsma:
    """Haagsma (2012) utility: U(x, y) = alpha_x ln(x - gamma_x) - alpha_y ln(gamma_y - y).

    A separable, quasi-concave utility whose good *x* is always inferior and
    becomes a Giffen good when income is high enough. Utility is defined for
    x > gamma_x and 0 <= y < gamma_y. At an interior optimum, Marshallian
    demand is

        x* = alpha_x (gamma_y py - I) / ((alpha_y - alpha_x) px)
             + alpha_y gamma_x / (alpha_y - alpha_x)

    so dx*/dI < 0 always, and dx*/dpx > 0 (Giffen) exactly when
    I > gamma_y * py. An optimum exists only while I < gamma_y * py +
    gamma_x * px: with more income the consumer can push y towards gamma_y and
    utility grows without bound.

    Reference: Haagsma, R. (2012). A convenient utility function with Giffen
    behaviour. *ISRN Economics*, 2012, 1-4. https://doi.org/10.5402/2012/608645

    Parameters
    ----------
    alpha_x : float
        Weight on good *x*. Must be positive and below *alpha_y*.
    alpha_y : float
        Weight on good *y*. Must exceed *alpha_x*.
    gamma_x : float
        Lower bound of good *x*. Must be positive.
    gamma_y : float
        Upper bound of good *y*. Must be positive.
    """

    alpha_x: float = 1.0
    alpha_y: float = 2.0
    gamma_x: float = 2.0
    gamma_y: float = 27.0

    def __post_init__(self) -> None:
        _require_positive(
            "Haagsma", alpha_x=self.alpha_x, alpha_y=self.alpha_y, gamma_x=self.gamma_x, gamma_y=self.gamma_y
        )
        if not self.alpha_x < self.alpha_y:
            raise InvalidParameterError("Haagsma: alpha_x must be smaller than alpha_y.")

    @property
    def utility_type(self) -> UtilityType:
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        dx = np.asarray(x, dtype=float) - self.gamma_x
        dy = self.gamma_y - np.asarray(y, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(
                (dx > 0) & (dy > 0),
                self.alpha_x * np.log(dx) - self.alpha_y * np.log(dy),
                np.nan,
            )
        return float(result) if result.ndim == 0 else result

    def lower_bounds(self) -> tuple[float, float]:
        """Return the minimum feasible (x, y)."""
        return (self.gamma_x, 0.0)

    def upper_bounds(self) -> tuple[float, float]:
        """Return the supremum of feasible (x, y); y must stay below gamma_y."""
        return (np.inf, self.gamma_y)

    def validate_budget(self, px: float, py: float, income: float) -> None:
        """Raise when utility is unbounded on the budget line, so no optimum exists."""
        ceiling = self.gamma_y * py + self.gamma_x * px
        if income >= ceiling:
            raise InvalidParameterError(
                f"Haagsma: income ({income:g}) must be below gamma_y*py + gamma_x*px ({ceiling:g}); "
                "otherwise y can approach gamma_y and utility has no maximum."
            )

    def demand(self, px: float, py: float, income: float) -> tuple[float, float]:
        """Closed-form interior Marshallian demand (x*, y*).

        Valid when the result satisfies x* > gamma_x and 0 <= y* < gamma_y.
        """
        spread = self.alpha_y - self.alpha_x
        x = self.alpha_x * (self.gamma_y * py - income) / (spread * px) + self.alpha_y * self.gamma_x / spread
        return x, (income - px * x) / py

    def is_giffen(self, px: float, py: float, income: float) -> bool:
        """Return whether x is a Giffen good: gamma_y*py < I < gamma_y*py + gamma_x*px."""
        return self.gamma_y * py < income < self.gamma_y * py + self.gamma_x * px

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []

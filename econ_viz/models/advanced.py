"""
Advanced utility function specifications.

This module provides two extension classes that sit on top of the standard
protocol:

* :class:`CustomUtility` — wrap any Python callable as a first-class model.
* :class:`MultiGoodCD` — N-good Cobb-Douglas with a ``freeze()`` projection
  down to the 2-D canvas.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..enums import UtilityType
from ..exceptions import InvalidParameterError
from ..utils.logging import get_logger

logger = get_logger(__name__)

_PROBE_SIZE = 4


def _probe(func: Callable, label: str) -> None:
    """Validate that *func* accepts and returns a NumPy array of shape (N, N).

    Parameters
    ----------
    func :
        Callable to probe with a random (N, N) mesh-grid pair.
    label :
        Human-readable name used in error messages.

    Raises
    ------
    ValueError
        If the function raises, returns a non-numeric type, or returns an
        array whose shape does not broadcast to the input shape.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 5.0, (_PROBE_SIZE, _PROBE_SIZE))
    Y = rng.uniform(0.1, 5.0, (_PROBE_SIZE, _PROBE_SIZE))

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = func(X, Y)
        result = np.asarray(result, dtype=float)
    except Exception as exc:
        raise ValueError(
            f"{label}: 自訂函數無法處理 NumPy 陣列運算。"
            f" 原始錯誤：{exc}"
        ) from exc

    if result.shape != X.shape and result.shape != ():
        raise ValueError(
            f"{label}: 自訂函數回傳形狀 {result.shape}，"
            f"預期 {X.shape} 或純量。"
        )


@dataclass
class CustomUtility:
    """Wrap an arbitrary callable as a standard ``UtilityFunction`` model.

    The callable is validated at construction time by probing it with a
    random NumPy mesh-grid.  Any function that cannot handle vectorised
    array arithmetic will be rejected immediately with a clear message.

    Parameters
    ----------
    func : Callable[[ndarray, ndarray], ndarray]
        A vectorised utility function ``f(x, y)`` returning element-wise
        utility values.  Must accept two arrays of identical shape and
        return an array of the same shape (or a scalar).
    name : str
        Display name used in log messages and ``repr``.

    Examples
    --------
    >>> import numpy as np
    >>> from econ_viz.models import CustomUtility
    >>> model = CustomUtility(func=lambda x, y: np.log(x) + np.log(y))
    >>> model(2.0, 3.0)
    1.791759469228327
    """

    func: Callable
    name: str = "custom"

    def __post_init__(self) -> None:
        _probe(self.func, label=repr(self.name))
        logger.debug("CustomUtility '%s' validated successfully.", self.name)

    @property
    def utility_type(self) -> UtilityType:
        """Always SMOOTH — the canvas draws generic contours."""
        return UtilityType.SMOOTH

    def __call__(self, x, y):
        return self.func(x, y)

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []

    def __repr__(self) -> str:
        return f"CustomUtility(name={self.name!r})"


class MultiGoodCD:
    """N-good Cobb-Douglas utility: U = x1^a1 * x2^a2 * … * xn^an.

    This class models preferences over *n* goods.  Because the canvas only
    renders 2-D diagrams, a :meth:`freeze` projection must be applied before
    passing the result to :meth:`~econ_viz.canvas.Canvas.add_utility`.

    Parameters
    ----------
    alphas : dict[str, float]
        Mapping of good names to their exponents, e.g.
        ``{'x': 0.3, 'y': 0.3, 'z': 0.4}``.  Exponents need not sum to 1.

    Raises
    ------
    InvalidParameterError
        If fewer than 2 goods are supplied, or if any exponent is
        non-positive.

    Examples
    --------
    >>> m = MultiGoodCD({'x': 0.3, 'y': 0.3, 'z': 0.4})
    >>> flat = m.freeze(z=10.0)
    >>> flat(2.0, 3.0)     # U(x=2, y=3, z=10)
    """

    def __init__(self, alphas: dict[str, float]) -> None:
        if len(alphas) < 2:
            raise InvalidParameterError(
                "MultiGoodCD requires at least 2 goods."
            )
        if any(a <= 0 for a in alphas.values()):
            raise InvalidParameterError(
                "All exponents in MultiGoodCD must be positive."
            )
        self._alphas: dict[str, float] = dict(alphas)
        logger.debug(
            "MultiGoodCD created: %s",
            ", ".join(f"{k}^{v}" for k, v in self._alphas.items()),
        )

    @property
    def utility_type(self) -> UtilityType:
        """Always SMOOTH."""
        return UtilityType.SMOOTH

    def evaluate(self, **goods: float) -> float:
        """Evaluate the full N-dimensional utility at a single bundle.

        Parameters
        ----------
        **goods :
            Keyword arguments mapping good names to quantities.
            All goods declared in *alphas* must be supplied.

        Returns
        -------
        float
            U = ∏ xᵢ^aᵢ.
        """
        missing = set(self._alphas) - set(goods)
        if missing:
            raise ValueError(f"Missing goods: {missing}")
        result = 1.0
        for name, alpha in self._alphas.items():
            result *= float(goods[name]) ** alpha
        return result

    def freeze(self, **fixed: float) -> CustomUtility:
        """Project the N-D function to 2-D by fixing all but two goods.

        The two *active* (unfrozen) variables are the keys of *alphas* that
        are not present in *fixed*, taken in insertion order.  The returned
        :class:`CustomUtility` object conforms to the ``UtilityFunction``
        protocol and can be passed directly to
        :meth:`~econ_viz.canvas.Canvas.add_utility`.

        Parameters
        ----------
        **fixed :
            Keyword arguments mapping good names to the constant quantities
            at which they are held.

        Returns
        -------
        CustomUtility
            A 2-D projection ``f(x, y)`` where the first active variable
            maps to the canvas x-axis and the second to the y-axis.

        Raises
        ------
        ValueError
            If the number of remaining (unfrozen) variables is not exactly 2.
        InvalidParameterError
            If a fixed good name is not in *alphas*.

        Examples
        --------
        >>> m = MultiGoodCD({'x': 0.3, 'y': 0.3, 'z': 0.4})
        >>> flat = m.freeze(z=10.0)
        >>> isinstance(flat, CustomUtility)
        True
        """
        unknown = set(fixed) - set(self._alphas)
        if unknown:
            raise InvalidParameterError(
                f"Unknown goods in freeze(): {unknown}. "
                f"Valid goods: {set(self._alphas)}"
            )

        active = [k for k in self._alphas if k not in fixed]
        if len(active) != 2:
            raise ValueError(
                "繪製 2D 圖表前，必須凍結變數直到只剩下 2 個活動變數。"
                f" 目前活動變數：{active}"
            )

        x_name, y_name = active[0], active[1]
        frozen_contribution = functools.reduce(
            lambda acc, item: acc * (item[1] ** self._alphas[item[0]]),
            fixed.items(),
            1.0,
        )
        ax = self._alphas[x_name]
        ay = self._alphas[y_name]

        label = (
            f"MultiGoodCD({x_name}^{ax} {y_name}^{ay}"
            + "".join(f" {k}={v}" for k, v in fixed.items())
            + ")"
        )

        def _projected(x, y, _k=frozen_contribution, _ax=ax, _ay=ay):
            return _k * (x ** _ax) * (y ** _ay)

        logger.info(
            "freeze() → active=(%s, %s), fixed=%s", x_name, y_name, fixed
        )
        return CustomUtility(func=_projected, name=label)

    def ray_slopes(self) -> list[float]:
        return []

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return []

    def __repr__(self) -> str:
        terms = " ".join(f"{k}^{v}" for k, v in self._alphas.items())
        return f"MultiGoodCD({terms})"

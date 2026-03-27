"""
Structural typing protocol for utility function models.

Any object satisfying :class:`UtilityFunction` can be passed to
:meth:`Canvas.add_utility` and will receive full rendering support
(contour lines, rays, kink markers) without inheritance coupling.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from ..enums import UtilityType


@runtime_checkable
class UtilityFunction(Protocol):
    """Structural interface that all utility models must satisfy.

    A conforming object is **callable** — ``model(x, y)`` evaluates the
    utility function element-wise — and exposes metadata used by the
    rendering layer.
    """

    @property
    def utility_type(self) -> UtilityType:
        """Qualitative shape category of the indifference map."""
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate U(x, y) element-wise."""
        ...

    def ray_slopes(self) -> list[float]:
        """Slopes of economically significant rays from the origin."""
        ...

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        """Return (x, y) coordinates of kink points on the given contour levels.

        Models whose ``utility_type`` is not ``KINKED`` should return an
        empty list.

        Parameters
        ----------
        levels : list[float]
            Utility values at which kink points should be computed.

        Returns
        -------
        list[tuple[float, float]]
            Sequence of ``(x, y)`` pairs, one per level.
        """
        ...

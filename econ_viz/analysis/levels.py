"""Indifference-curve level computation strategies.

This module decouples the *placement* of contour levels from both the
utility model and the canvas, allowing callers to choose a spacing
strategy independently.

Typical usage with an equilibrium anchor::

    eq = solve(model, px, py, income)
    lvls = levels.around(eq.utility, n=5)
    cvs.add_utility(model, levels=lvls)
"""

from __future__ import annotations

from ..contours import around_anchor_levels, percentile_levels


def around(anchor: float, n: int = 5, spread: float = 0.5) -> list[float]:
    """Compute *n* evenly spaced utility levels centred on *anchor*.

    The anchor value (typically the equilibrium utility) is always
    included as one of the returned levels, ensuring that any plotted
    equilibrium point lies exactly on a visible indifference curve.

    Parameters
    ----------
    anchor : float
        Reference utility level placed at the centre of the range.
    n : int
        Total number of levels to produce (must be >= 1).
    spread : float
        Half-width of the range expressed as a fraction of *anchor*.
        Levels span ``[anchor * (1 - spread), anchor * (1 + spread)]``.

    Returns
    -------
    list[float]
        Sorted utility levels with *anchor* at the midpoint.
    """
    return around_anchor_levels(anchor=anchor, n=n, spread=spread)


def percentile(Z: np.ndarray, n: int = 5, lo: float = 20, hi: float = 80) -> list[float]:
    """Compute *n* levels by percentile of a pre-computed utility surface.

    This is the strategy used when no equilibrium anchor is available.

    Parameters
    ----------
    Z : numpy.ndarray
        2-D array of utility values (as returned by
        :meth:`~econ_viz.canvas.layers.Layer.compute_contour`).
    n : int
        Number of levels.
    lo, hi : float
        Lower and upper percentile bounds (0–100).

    Returns
    -------
    list[float]
        Utility levels at the requested percentiles.
    """
    return percentile_levels(Z=Z, n=n, lo=lo, hi=hi)

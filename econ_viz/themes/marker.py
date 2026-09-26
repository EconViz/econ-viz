"""Marker: how one point is drawn (colour, size, and shape)."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.markers import MarkerStyle

from ..exceptions import InvalidParameterError
from .opacity import check_opacity


@dataclass(frozen=True)
class Marker:
    """How to draw one point marker. Fields left as ``None`` keep the default.

    Parameters
    ----------
    color : str, optional
        Any Matplotlib colour; sets both the fill and the edge.
    size : float, optional
        Marker size in points.
    shape : str, optional
        Any Matplotlib marker, e.g. ``"o"``, ``"s"``, ``"^"``, ``"D"``, ``"*"``.
    opacity : float, optional
        From 0 (transparent) to 1 (opaque).
    """

    color: str | None = None
    size: float | None = None
    shape: str | None = None
    opacity: float | None = None

    def __post_init__(self) -> None:
        check_opacity("Marker", self.opacity)
        if self.size is not None and not self.size > 0:
            raise InvalidParameterError(f"Marker size must be positive, got {self.size!r}")
        if self.shape is not None:
            try:
                MarkerStyle(self.shape)
            except ValueError:
                raise InvalidParameterError(f"invalid Marker shape {self.shape!r}") from None

    def merged_over(self, base: Marker | None) -> Marker:
        """Return this marker with unset fields taken from *base*."""
        if base is None:
            return self
        return Marker(
            color=self.color if self.color is not None else base.color,
            size=self.size if self.size is not None else base.size,
            shape=self.shape if self.shape is not None else base.shape,
            opacity=self.opacity if self.opacity is not None else base.opacity,
        )

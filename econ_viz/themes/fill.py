"""Fill: how one shaded region is drawn (colour and opacity)."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.colors import is_color_like

from ..exceptions import InvalidParameterError


@dataclass(frozen=True)
class Fill:
    """How to shade one region. Fields left as ``None`` keep the default.

    Parameters
    ----------
    color : str, optional
        Any Matplotlib colour. ``None`` follows the outline's colour.
    alpha : float, optional
        Opacity from 0 (clear) to 1 (solid).
    """

    color: str | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.color is not None and not is_color_like(self.color):
            raise InvalidParameterError(f"invalid Fill color {self.color!r}")
        if self.alpha is not None and not 0 <= self.alpha <= 1:
            raise InvalidParameterError(f"Fill alpha must be between 0 and 1, got {self.alpha!r}")

    def merged_over(self, base: Fill | None) -> Fill:
        """Return this fill with unset fields taken from *base*."""
        if base is None:
            return self
        return Fill(
            color=self.color if self.color is not None else base.color,
            alpha=self.alpha if self.alpha is not None else base.alpha,
        )

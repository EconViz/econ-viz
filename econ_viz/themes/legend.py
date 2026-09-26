"""Legend: where a legend goes and how it looks."""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import LegendPosition
from ..exceptions import InvalidParameterError


@dataclass(frozen=True)
class Legend:
    """How to draw a legend. Fields left as ``None`` keep the default.

    Parameters
    ----------
    position : LegendPosition or str, optional
        ``"auto"`` picks the inside corner that covers the least of the
        diagram and moves outside to the right when every corner covers
        something. Inside corners: ``"upper right"``, ``"upper left"``,
        ``"lower left"``, ``"lower right"``. Outside the plot area: ``"top"``,
        ``"bottom"``, ``"left"``, ``"right"``.
    fontsize : float, optional
        Font size in points.
    frame : bool, optional
        Draw a frame around the legend.
    columns : int, optional
        Number of columns.
    visible : bool, optional
        ``False`` draws no legend.
    """

    position: LegendPosition | str | None = None
    fontsize: float | None = None
    frame: bool | None = None
    columns: int | None = None
    visible: bool | None = None

    def __post_init__(self) -> None:
        if self.position is not None:
            try:
                object.__setattr__(self, "position", LegendPosition(self.position))
            except ValueError:
                choices = ", ".join(p.value for p in LegendPosition)
                raise InvalidParameterError(
                    f"invalid Legend position {self.position!r}; choose: {choices}"
                ) from None
        if self.fontsize is not None and not self.fontsize > 0:
            raise InvalidParameterError(f"Legend fontsize must be positive, got {self.fontsize!r}")
        if self.columns is not None and (int(self.columns) != self.columns or self.columns < 1):
            raise InvalidParameterError(f"Legend columns must be a positive integer, got {self.columns!r}")

    def merged_over(self, base: Legend | None) -> Legend:
        """Return this legend with unset fields taken from *base*."""
        if base is None:
            return self
        fields = ("position", "fontsize", "frame", "columns", "visible")
        return Legend(**{
            field: getattr(self, field) if getattr(self, field) is not None else getattr(base, field)
            for field in fields
        })

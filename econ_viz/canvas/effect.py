"""Effect: colour, position, and label of one decomposition effect arrow."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..enums import LabelPosition
from ..themes.label import Label, split_label, to_position
from ..themes.opacity import check_opacity


@dataclass(frozen=True)
class Effect:
    """How to show the substitution or income effect in a decomposition.

    Parameters
    ----------
    color : str, optional
        Colour of the effect arrow, its range arrow below the x-axis, its
        label, and its legend entry.
    y : float, optional
        Height of the range arrow below the x-axis, as a fraction of the axes
        height (negative is below the axis). Used with ``show_x_projections``.
    label : str or Label, optional
        Text drawn next to the arrow, or a :class:`Label` that also sets its
        position, offset, font size, colour, and visibility (default
        ``theme.effect_label``).
    label_position : LabelPosition or str
        Side of the arrow's middle where the text goes. For the horizontal
        range arrows below the x-axis, ``"left"`` / ``"right"`` place it past
        the arrow's left or right end. ``"above"`` and ``"below"`` are
        accepted as aliases for ``"top"`` and ``"bottom"``. Shorthand for
        ``Label(position=...)``; a Label position wins.
    label_offset : float
        Distance between the arrow and the text, in points. Shorthand for
        ``Label(offset=...)``; a Label offset wins.
    opacity : float, optional
        From 0 (transparent) to 1 (opaque) for the arrow, its range arrow, and
        its label. A Stroke or Label opacity wins for its own part.
    """

    color: str | None = None
    y: float | None = None
    label: str | Label | None = None
    label_position: LabelPosition | str = LabelPosition.RIGHT
    label_offset: float = 4.0
    opacity: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_position", to_position(self.label_position, what="Effect label_position"))
        check_opacity("Effect", self.opacity)

    def resolved_label(self, default: Label) -> tuple[str | None, Label]:
        """Return the label text and its style: Label fields, then the shorthand, then *default*."""
        text, style = split_label(self.label, default)
        own = self.label if isinstance(self.label, Label) else Label()
        return text, replace(
            style,
            position=own.position or self.label_position,
            offset=own.offset if own.offset is not None else self.label_offset,
        )

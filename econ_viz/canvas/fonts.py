"""Per-figure font configuration that leaves Matplotlib's global rcParams untouched."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text

from ..exceptions import InvalidParameterError
from .stroke import tag_attr

GENERIC_FAMILIES = frozenset({"serif", "sans-serif", "monospace", "cursive", "fantasy"})
MATH_FONTS = ("dejavusans", "dejavuserif", "cm", "stix", "stixsans")


def resolve_font(font: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Validate a font family (or fallback list) and return it as a tuple.

    At least one family must be a generic family or installed on the system.
    """
    if font is None:
        return None
    families = (font,) if isinstance(font, str) else tuple(font)
    if not families or not all(isinstance(f, str) and f.strip() for f in families):
        raise InvalidParameterError("font must be a family name or a non-empty list of family names")
    if not any(_is_available(f) for f in families):
        raise InvalidParameterError(
            f"font not found: {', '.join(families)}. Install it, add a fallback, "
            "or use a generic family such as 'serif', 'sans-serif', or 'monospace'."
        )
    return families


def resolve_math_font(math_font: str | None) -> str | None:
    """Validate a Matplotlib math font set name."""
    if math_font is None or math_font in MATH_FONTS:
        return math_font
    raise InvalidParameterError(f"invalid math_font {math_font!r}; choose: {', '.join(MATH_FONTS)}")


def _is_available(family: str) -> bool:
    if family in GENERIC_FAMILIES:
        return True
    try:
        findfont(FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        return False
    return True


class FontApplier(Artist):
    """Invisible artist that sets the fonts of every text in its figure right before drawing.

    Drawing first (lowest zorder) means text added after construction — later layers,
    legends, or a notebook's automatic display — picks up the font too.
    """

    zorder = float("-inf")

    def __init__(self, families: tuple[str, ...] | None, math_font: str | None = None):
        super().__init__()
        self.families = list(families) if families else None
        self.math_font = math_font

    def draw(self, renderer) -> None:
        for text in self.figure.findobj(Text):
            if self.families:
                text.set_fontfamily(self.families)
                tag_attr(text, "_ev_font", self.families)
            if self.math_font:
                text.set_math_fontfamily(self.math_font)
        self.stale = False

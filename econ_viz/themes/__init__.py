"""Built-in colour themes for economic diagrams.

The default theme uses a color-blind-friendly palette.

References
----------
- https://gist.github.com/thriveth/8560036
- https://www.tandfonline.com/doi/abs/10.1080/00220485.1996.10844911
"""

from typing import TypedDict

from ..enums import ArrowStyle, LineStyle
from .label import Label
from .marker import Marker
from .stroke import Stroke
from .theme import Theme

COLORBLIND_CYCLE_RGB: tuple[tuple[int, int, int], ...] = (
    (55, 126, 184),  # blue
    (255, 127, 0),  # orange
    (77, 175, 74),  # green
    (247, 129, 191),  # pink
    (166, 86, 40),  # brown
    (152, 78, 163),  # purple
    (153, 153, 153),  # gray
    (228, 26, 28),  # red
    (222, 222, 0),  # yellow
)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


COLORBLIND_CYCLE_HEX: tuple[str, ...] = tuple(_rgb_to_hex(rgb) for rgb in COLORBLIND_CYCLE_RGB)

(
    _CB_BLUE,
    _CB_ORANGE,
    _CB_GREEN,
    _CB_PINK,
    _CB_BROWN,
    _CB_PURPLE,
    _CB_GRAY,
    _CB_RED,
    _CB_YELLOW,
) = COLORBLIND_CYCLE_HEX


class _ColorblindBase(TypedDict):
    axis_color: str
    label_color: str
    ic_color: str
    ic_linewidth: float
    path_color: str
    budget_color: str
    eq_color: str
    ray_color: str
    kink_color: str
    sub_effect_color: str
    inc_effect_color: str
    compensated_budget_color: str


_COLORBLIND_BASE: _ColorblindBase = {
    "axis_color": "#222222",
    "label_color": "#222222",
    "ic_color": _CB_BLUE,
    "ic_linewidth": 1.8,
    "path_color": _CB_GREEN,
    "budget_color": _CB_PURPLE,
    "eq_color": _CB_RED,
    "ray_color": _CB_GRAY,
    "kink_color": _CB_BROWN,
    "sub_effect_color": _CB_ORANGE,
    "inc_effect_color": _CB_GREEN,
    "compensated_budget_color": "#777777",
}

default = Theme(name="default", **_COLORBLIND_BASE)
colorblind = Theme(name="colorblind", **_COLORBLIND_BASE)

nord = Theme(
    name="nord",
    axis_color="#2E3440",
    label_color="#2E3440",
    ic_color="#88C0D0",
    path_color="#A3BE8C",
    budget_color="#5E81AC",
    budget_fill_alpha=0.10,
    eq_color="#BF616A",
    ray_color="#4C566A",
    ray_linewidth=1.0,
    kink_color="#88C0D0",
)

# ------------------------------------------------------------------
# Purpose-driven themes (#76): journal figures, black-and-white
# printing, classroom slides, and dark interfaces.
# ------------------------------------------------------------------

# Restrained typography and thinner lines for journal figures. Uses the
# fully transparent (print-safe) background that's already the Theme default.
paper = Theme(
    name="paper",
    axis_color="#1A1A1A",
    label_color="#1A1A1A",
    ic_color="#345995",
    ic_linewidth=1.2,
    path_color="#4B7355",
    path_linewidth=1.4,
    budget_color="#5C5C5C",
    budget_linewidth=1.0,
    budget_fill_alpha=0.06,
    eq_color="#8C2F39",
    eq_markersize=3.0,
    ray_color="#8C8C8C",
    ray_linewidth=0.6,
    kink_color="#6B4226",
    sub_effect_color="#B8621B",
    inc_effect_color="#4B7355",
    effect_arrow_linewidth=1.1,
    compensated_budget_color="#8C8C8C",
    compensated_budget_linewidth=1.0,
    subsistence_color="#8C8C8C",
    subsistence_linewidth=0.6,
    contract_color="#1A1A1A",
    contract_linewidth=0.9,
    core_color="#8C2F39",
    core_linewidth=2.0,
    price_color="#1A1A1A",
    price_linewidth=0.9,
    walrasian_color="#345995",
    walrasian_markersize=7.0,
    # Transparent (print-safe) background is the Theme default; not set here.
    label_scale=0.9,
)


class _MonochromeTheme(Theme):
    """Differentiates lines by style and points by shape, not colour, for black-and-white printing."""

    @property
    def ic_stroke(self) -> Stroke:
        return Stroke(width=self.ic_linewidth, style=LineStyle.SOLID, color=self.ic_color)

    @property
    def budget_stroke(self) -> Stroke:
        return Stroke(width=self.budget_linewidth, style=LineStyle.DASHED, color=self.budget_color)

    @property
    def ray_stroke(self) -> Stroke:
        return Stroke(width=self.ray_linewidth, style=LineStyle.DOTTED, color=self.ray_color)

    @property
    def path_stroke(self) -> Stroke:
        return Stroke(width=self.path_linewidth, style=LineStyle.DASHDOT, color=self.path_color)

    @property
    def substitution_stroke(self) -> Stroke:
        return Stroke(
            width=self.effect_arrow_linewidth,
            style=LineStyle.SOLID,
            color=self.sub_effect_color,
            arrow=ArrowStyle.TRIANGLE,
        )

    @property
    def income_stroke(self) -> Stroke:
        return Stroke(
            width=self.effect_arrow_linewidth,
            style=LineStyle.DASHED,
            color=self.inc_effect_color,
            arrow=ArrowStyle.SIMPLE,
        )

    @property
    def eq_marker(self) -> Marker:
        return Marker(color=self.eq_color, size=self.eq_markersize, shape="o")

    @property
    def point_marker(self) -> Marker:
        return Marker(color=self.eq_color, size=self.eq_markersize, shape="s")

    @property
    def kink_marker(self) -> Marker:
        return Marker(color=self.kink_color, size=4.0, shape="^")

    @property
    def path_marker(self) -> Marker:
        return Marker(size=max(self.eq_markersize - 1, 3), shape="D")

    @property
    def core_marker(self) -> Marker:
        return Marker(color=self.core_color, shape="P")

    @property
    def endowment_marker(self) -> Marker:
        return Marker(color=self.eq_color, size=max(self.eq_markersize, 6.0), shape="s")

    @property
    def walrasian_marker(self) -> Marker:
        return Marker(color=self.walrasian_color, size=self.walrasian_markersize, shape="D")


monochrome = _MonochromeTheme(
    name="monochrome",
    axis_color="#000000",
    label_color="#000000",
    ic_color="#000000",
    path_color="#000000",
    budget_color="#000000",
    eq_color="#000000",
    ray_color="#555555",
    kink_color="#000000",
    sub_effect_color="#000000",
    inc_effect_color="#000000",
    compensated_budget_color="#777777",
    subsistence_color="#888888",
    contract_color="#000000",
    core_color="#000000",
    price_color="#555555",
    walrasian_color="#000000",
    walrasian_markersize=11.0,
)


class _PresentationTheme(Theme):
    """Larger text and a stronger line/marker hierarchy for projected slides."""

    @property
    def title_label(self) -> Label:
        return Label(fontsize=22)

    @property
    def box_label(self) -> Label:
        return Label(fontsize=16)


presentation = _PresentationTheme(
    name="presentation",
    axis_color="#1A1A1A",
    label_color="#1A1A1A",
    ic_color="#1F77B4",
    ic_linewidth=2.6,
    path_color="#2CA02C",
    path_linewidth=2.8,
    budget_color="#9467BD",
    budget_linewidth=2.2,
    eq_color="#D62728",
    eq_markersize=7.0,
    ray_color="#7F7F7F",
    ray_linewidth=1.2,
    kink_color="#8C564B",
    sub_effect_color="#FF7F0E",
    inc_effect_color="#2CA02C",
    effect_arrow_linewidth=2.2,
    compensated_budget_color="#7F7F7F",
    compensated_budget_linewidth=2.0,
    subsistence_color="#7F7F7F",
    subsistence_linewidth=1.2,
    contract_color="#1A1A1A",
    contract_linewidth=2.0,
    core_color="#D62728",
    core_linewidth=4.0,
    price_color="#1A1A1A",
    price_linewidth=1.8,
    walrasian_color="#1F77B4",
    walrasian_markersize=14.0,
    label_scale=1.4,
)

dark = Theme(
    name="dark",
    axis_color="#E0E0E0",
    label_color="#E0E0E0",
    background_color="#1E1E1E",
    ic_color="#4FC3F7",
    path_color="#81C784",
    budget_color="#CE93D8",
    eq_color="#EF5350",
    ray_color="#B0BEC5",
    kink_color="#FFB74D",
    sub_effect_color="#FFB74D",
    inc_effect_color="#81C784",
    compensated_budget_color="#B0BEC5",
    subsistence_color="#B0BEC5",
    contract_color="#E0E0E0",
    core_color="#EF5350",
    price_color="#E0E0E0",
    walrasian_color="#4FC3F7",
)

__all__ = [
    "Theme",
    "COLORBLIND_CYCLE_RGB",
    "COLORBLIND_CYCLE_HEX",
    "default",
    "colorblind",
    "nord",
    "paper",
    "monochrome",
    "presentation",
    "dark",
]

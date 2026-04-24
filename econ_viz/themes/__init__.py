"""Built-in colour themes for economic diagrams.

The default theme uses a color-blind-friendly palette.

References
----------
- https://gist.github.com/thriveth/8560036
- https://www.tandfonline.com/doi/abs/10.1080/00220485.1996.10844911
"""

from .theme import Theme

COLORBLIND_CYCLE_RGB: tuple[tuple[int, int, int], ...] = (
    (55, 126, 184),   # blue
    (255, 127, 0),    # orange
    (77, 175, 74),    # green
    (247, 129, 191),  # pink
    (166, 86, 40),    # brown
    (152, 78, 163),   # purple
    (153, 153, 153),  # gray
    (228, 26, 28),    # red
    (222, 222, 0),    # yellow
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

_COLORBLIND_BASE = dict(
    axis_color="#222222",
    label_color="#222222",
    ic_color=_CB_BLUE,
    ic_linewidth=1.8,
    path_color=_CB_GREEN,
    budget_color=_CB_PURPLE,
    eq_color=_CB_RED,
    ray_color=_CB_GRAY,
    kink_color=_CB_BROWN,
    sub_effect_color=_CB_ORANGE,
    inc_effect_color=_CB_GREEN,
    compensated_budget_color="#777777",
)

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

__all__ = [
    "Theme",
    "COLORBLIND_CYCLE_RGB",
    "COLORBLIND_CYCLE_HEX",
    "default",
    "colorblind",
    "nord",
]

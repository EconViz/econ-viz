"""Built-in colour themes for economic diagrams."""

from .theme import Theme

default = Theme(name="default")

nord = Theme(
    name="nord",
    axis_color="#2E3440",
    label_color="#2E3440",
    ic_color="#88C0D0",
    budget_color="#5E81AC",
    budget_fill_alpha=0.10,
    eq_color="#BF616A",
    ray_color="#4C566A",
    ray_linewidth=1.0,
    kink_color="#88C0D0",
)

__all__ = ["Theme", "default", "nord"]

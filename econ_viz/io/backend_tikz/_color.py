"""Named-color registry for TikZ output.

Matplotlib hands the renderer raw RGB(A) tuples on every ``draw_path``
call.  TikZ, on the other hand, references colors by *name* through the
``xcolor`` package.  This module bridges the two: each unique RGB triple
seen by the renderer is assigned a short, stable identifier, and every
identifier becomes a single ``\\definecolor`` line in the emitted preamble.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[float, float, float]


@dataclass
class ColorRegistry:
    _names: dict[tuple[int, int, int], str] = field(default_factory=dict)

    def register(self, rgb) -> str:
        r, g, b = (float(c) for c in tuple(rgb)[:3])
        key = (_byte(r), _byte(g), _byte(b))
        name = self._names.get(key)
        if name is None:
            name = f"evcolor{len(self._names)}"
            self._names[key] = name
        return name

    def definitions(self) -> list[str]:
        return [
            rf"\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}"
            for (r, g, b), name in self._names.items()
        ]


def _byte(c: float) -> int:
    return max(0, min(255, round(c * 255)))

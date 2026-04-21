"""Assemble a :class:`TikzRenderer`'s output into a LaTeX source file."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderer import TikzRenderer


_STANDALONE_PREAMBLE = r"""\documentclass[border=8pt]{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\begin{document}
"""

_STANDALONE_POSTAMBLE = r"""\end{document}
"""


def assemble(renderer: "TikzRenderer", *, standalone: bool = True) -> str:
    """Serialise ``renderer``'s commands into a LaTeX string.

    Parameters
    ----------
    renderer
        A :class:`TikzRenderer` that has already consumed ``fig.draw(...)``.
    standalone
        If ``True`` (default) wrap the output in a full ``standalone``
        document that compiles by itself.  If ``False``, emit only the
        ``\\definecolor`` lines and ``tikzpicture`` environment — suitable
        for ``\\input`` from a parent document.
    """
    color_defs = renderer.colors.definitions()
    parts: list[str] = []
    if color_defs:
        parts.extend(color_defs)
        parts.append("")
    parts.append(r"\begin{tikzpicture}")
    parts.extend(renderer.commands)
    parts.append(r"\end{tikzpicture}")
    body = "\n".join(parts) + "\n"

    if not standalone:
        return body
    return _STANDALONE_PREAMBLE + body + _STANDALONE_POSTAMBLE

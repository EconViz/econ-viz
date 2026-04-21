"""Pure-TikZ export backend for ``econ_viz``.

This package implements a custom Matplotlib ``RendererBase`` that
converts figures produced by :class:`~econ_viz.Canvas` into self-contained
TikZ source.  The output compiles under plain ``pdflatex`` with only
``\\usepackage{tikz}`` (or the ``standalone`` document class), requiring
no ``pgfplots`` or other auxiliary packages.

Public entry points
-------------------
:func:`figure_to_tikz`
    Render a :class:`matplotlib.figure.Figure` to a TikZ string.
:func:`save_tikz`
    Export a :class:`matplotlib.figure.Figure` to a ``.tex`` file.
:class:`TikzRenderer`
    The low-level renderer, exposed for advanced use (custom assembly,
    inspection of emitted commands, alternative document wrappers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from matplotlib.figure import Figure as MplFigure

from ._document import assemble
from .renderer import TikzRenderer


def save_tikz(
    fig: MplFigure,
    path: Union[str, Path],
    *,
    scale: Optional[float] = None,
    standalone: bool = True,
) -> None:
    """Export ``fig`` to a TikZ ``.tex`` file.

    Parameters
    ----------
    fig
        The Matplotlib figure to export.
    path
        Destination ``.tex`` file path.  The parent directory must exist.
    scale
        Centimetres per pixel in the emitted TikZ.  Defaults to
        ``1.25 / fig.dpi``, so a 6 inch wide figure exports at about
        7.5 cm wide.
    standalone
        If ``True`` (default) emit a compilable document.  If ``False``
        emit only the ``\\definecolor`` lines + ``tikzpicture`` environment
        so the output can be ``\\input`` from a parent document.
    """
    text = figure_to_tikz(fig, scale=scale, standalone=standalone)
    Path(path).write_text(text, encoding="utf-8")


def figure_to_tikz(
    fig: MplFigure,
    *,
    scale: Optional[float] = None,
    standalone: bool = True,
) -> str:
    """Render ``fig`` to TikZ source and return it as a string."""
    # Matplotlib computes bounding boxes lazily; force a layout pass so
    # text placements and tick positions are up to date before we draw
    # into our renderer.
    fig.canvas.draw()

    width_px, height_px = fig.bbox.width, fig.bbox.height
    renderer = TikzRenderer(
        width_px,
        height_px,
        dpi=fig.get_dpi(),
        scale=scale,
    )
    fig.draw(renderer)

    return assemble(renderer, standalone=standalone)


__all__ = ["figure_to_tikz", "save_tikz", "TikzRenderer"]

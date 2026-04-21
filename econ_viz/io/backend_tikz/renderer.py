"""Matplotlib ``RendererBase`` subclass that emits pure TikZ."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backend_bases import RendererBase

from ._color import ColorRegistry
from ._path import dash_spec, path_to_polylines
from ._text import sanitize_text

_DEFAULT_CM_PER_INCH = 1.25


class TikzRenderer(RendererBase):
    """Intercept Matplotlib drawing calls and emit TikZ commands.

    Parameters
    ----------
    width, height : float
        Canvas size in pixels (matches ``fig.bbox``).
    dpi : float
        Figure DPI; used by ``points_to_pixels`` and as the default scale.
    scale : float
        Centimetres per pixel in the emitted TikZ.  The default of
        ``1.25 / dpi`` maps a 6 inch wide figure to 7.5 cm.
    """

    def __init__(
        self,
        width: float,
        height: float,
        *,
        dpi: float = 100.0,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.width = float(width)
        self.height = float(height)
        self.dpi = float(dpi)
        self.scale = float(scale if scale is not None else _DEFAULT_CM_PER_INCH / self.dpi)
        self.colors = ColorRegistry()
        self._commands: list[str] = []

    # ------------------------------------------------------------------
    # RendererBase required API
    # ------------------------------------------------------------------

    def get_canvas_width_height(self):
        return self.width, self.height

    def points_to_pixels(self, points):
        return points * self.dpi / 72.0

    def get_text_width_height_descent(self, s, prop, ismath):
        size = prop.get_size_in_points() if prop is not None else 10.0
        return 0.6 * size * max(len(s), 1), size, 0.2 * size

    def flipy(self):
        """Keep text coordinates in Matplotlib's bottom-left display space."""
        return False

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def draw_path(self, gc, path, transform, rgbFace=None):
        filled = rgbFace is not None
        for poly in path_to_polylines(path, transform, filled=filled):
            if len(poly) < 2:
                continue
            self._emit_path(gc, poly, rgbFace)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if not s:
            return
        is_axis_label = mtext is not None and getattr(mtext, "_ev_axis_label", None) in ("x", "y")
        body = sanitize_text(s, ismath=bool(ismath))
        if is_axis_label:
            inner = body
            if inner.startswith("$") and inner.endswith("$") and len(inner) >= 2:
                inner = inner[1:-1]
            body = rf"$\text{{\sffamily {inner}}}$"
        elif ismath and not (body.startswith("$") and body.endswith("$")):
            body = f"${body}$"
        color = self.colors.register(gc.get_rgb()[:3])
        size_pt = prop.get_size_in_points() if prop is not None else 10.0
        opts = [
            "anchor=base west",
            f"rotate={angle:.2f}",
            f"text={color}",
            "inner sep=0pt",
            f"font=\\fontsize{{{size_pt:.2f}}}{{{size_pt * 1.2:.2f}}}\\selectfont",
        ]
        self._commands.append(
            rf"\node[{','.join(opts)}] at "
            rf"({self._tx(x):.4f},{self._ty(y):.4f}) {{{body}}};"
        )

    def draw_image(self, gc, x, y, im, transform=None):
        # Raster images have no TikZ analogue in the pure-vector output
        # this backend is designed for.  Silently skip to avoid crashing
        # on figures that embed bitmaps (e.g. matplotlib logos).
        return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tx(self, x: float) -> float:
        return x * self.scale

    def _ty(self, y: float) -> float:
        return y * self.scale

    def _emit_path(self, gc, poly: np.ndarray, rgbFace) -> None:
        pts = " -- ".join(
            f"({self._tx(x):.4f},{self._ty(y):.4f})" for x, y in poly
        )
        stroke_name = self.colors.register(gc.get_rgb()[:3])
        lw_pt = gc.get_linewidth()
        alpha = gc.get_alpha()
        _, dash_list = gc.get_dashes()

        opts = [f"color={stroke_name}", f"line width={lw_pt:.2f}pt"]
        if alpha is not None and alpha < 1.0:
            opts.append(f"opacity={alpha:.3f}")
        if dash_list is not None and len(dash_list) > 0:
            opts.append(dash_spec(dash_list))

        if rgbFace is not None:
            fill_rgba = tuple(rgbFace)
            fill_name = self.colors.register(fill_rgba[:3])
            opts.append(f"fill={fill_name}")
            if len(fill_rgba) >= 4 and fill_rgba[3] < 1.0:
                opts.append(f"fill opacity={fill_rgba[3]:.3f}")
            cmd = r"\filldraw"
        else:
            cmd = r"\draw"
        self._commands.append(f"{cmd}[{','.join(opts)}] {pts};")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def commands(self) -> list[str]:
        return list(self._commands)

"""Core Animator class for GIF export via the factory pattern."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from econ_viz.canvas.base import Canvas
    from econ_viz.canvas.figure import Figure

    AnyFigure = Union[Canvas, Figure]


def _require_pillow() -> None:
    """Raise a clear ImportError if Pillow is not installed."""
    try:
        import PIL  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF export.\n"
            "Install it with:  pip install econ-viz[animation]\n"
            "              or:  pip install Pillow"
        ) from exc


def _figure_to_pil(fig, dpi: int):
    """Render a matplotlib Figure to a PIL RGBA Image."""
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy().convert("RGBA")


class Animator:
    """Generate a GIF animation by sweeping a parameter across a range.

    This class follows the **factory pattern**: for each frame value the
    caller-supplied *figure_factory* creates a fresh :class:`~econ_viz.canvas.base.Canvas`
    (or :class:`~econ_viz.canvas.figure.Figure`).  The Animator renders
    every frame to an in-memory image and stitches them into a GIF using
    Pillow — no ffmpeg required.

    Parameters
    ----------
    figure_factory : Callable[[float], Canvas | Figure]
        A function that accepts a single numeric frame value and returns a
        fully populated Canvas or Figure.  Write this exactly as you would
        write a normal econ_viz plotting script; the Animator handles
        opening and closing matplotlib figures automatically.
    frames : array-like of float
        Ordered sequence of parameter values — one element per animation
        frame (e.g. ``np.linspace(1.0, 10.0, 50)``).

    Examples
    --------
    >>> import numpy as np
    >>> from econ_viz import Canvas
    >>> from econ_viz.models import CobbDouglas
    >>> from econ_viz.animation import Animator
    >>>
    >>> def draw(p1: float) -> Canvas:
    ...     c = Canvas(x_max=10, y_max=10, x_label="X_1", y_label="X_2")
    ...     c.add_budget(px=p1, py=2.0, income=20.0)
    ...     c.add_utility(CobbDouglas(0.5, 0.5), levels=5)
    ...     return c
    >>>
    >>> Animator(draw, frames=np.linspace(1, 8, 40)).save("budget_sweep.gif", fps=12)
    """

    def __init__(
        self,
        figure_factory: Callable[[float], AnyFigure],
        frames: Sequence[float] | np.ndarray,
    ) -> None:
        self._factory = figure_factory
        self._frames: np.ndarray = np.asarray(frames, dtype=float)
        if self._frames.ndim != 1 or len(self._frames) == 0:
            raise ValueError("`frames` must be a non-empty 1-D array-like.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
        fps: int = 10,
        dpi: int = 150,
        loop: int = 0,
    ) -> None:
        """Render all frames and write the result to *path* as a GIF.

        Parameters
        ----------
        path : str or Path
            Destination file path, e.g. ``"price_sweep.gif"``.
        fps : int
            Frames per second.  Default ``10``.
        dpi : int
            Dots-per-inch used when rasterising each frame.  Lower values
            produce smaller files; ``150`` is a good balance for web use.
            Default ``150``.
        loop : int
            Number of GIF loops.  ``0`` means loop forever (default).
        """
        _require_pillow()
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        images = self._render_frames(dpi=dpi, plt=plt)
        self._write_gif(images, path=Path(path), fps=fps, loop=loop)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_frames(self, dpi: int, plt) -> list:
        """Call the factory for every frame value and collect PIL images."""
        images = []
        for val in self._frames:
            canvas_or_fig = self._factory(float(val))
            mpl_fig = canvas_or_fig.fig
            images.append(_figure_to_pil(mpl_fig, dpi=dpi))
            plt.close(mpl_fig)
        return images

    @staticmethod
    def _write_gif(
        images: list,
        path: Path,
        fps: int,
        loop: int,
    ) -> None:
        """Stitch PIL images into a GIF file using Pillow."""
        path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = int(1000 / max(fps, 1))
        images[0].save(
            str(path),
            save_all=True,
            append_images=images[1:],
            loop=loop,
            duration=duration_ms,
            optimize=False,
        )

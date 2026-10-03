"""Showing and exporting the figure."""

from __future__ import annotations

from utility_viz.core.canvas._state import CanvasState
from utility_viz.core.export import save_figure
from utility_viz.utils.logging import get_logger

logger = get_logger(__name__)


class OutputMixin(CanvasState):
    """``show`` and ``save`` for :class:`~utility_viz.core.canvas.base.Canvas`."""

    def show(self) -> None:
        """Display the figure in an interactive matplotlib window."""
        self.fig.show()

    def save(self, path: str, **kwargs) -> None:
        """Export the figure to disk and release matplotlib resources.

        The output format is inferred from the file extension via
        :class:`~utility_viz.enums.ExportFormat`.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"plot.png"``, ``"plot.svg"``,
            ``"plot.tex"``).
        **kwargs
            Forwarded to the underlying save function. TikZ exports accept
            ``tikz_scale`` and ``tikz_standalone``.
        """
        logger.info("Exporting figure to %s (dpi=%s)", path, self.dpi)
        kwargs.setdefault("transparent", self.theme.background_color is None)
        save_figure(
            self.fig,
            path=path,
            dpi=self.dpi,
            close=self._owns_figure,
            **kwargs,
        )

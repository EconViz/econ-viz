"""Shared figure-export helpers."""

from __future__ import annotations

from matplotlib.figure import Figure as MplFigure

from ..enums import ExportFormat
from ..exceptions import ExportError


def save_figure(
    fig: MplFigure,
    *,
    path: str,
    dpi: int,
    close: bool = False,
    unsupported_as_value_error: bool = False,
    **kwargs,
) -> None:
    """Save a matplotlib figure with consistent format validation and errors."""
    try:
        ExportFormat.from_path(path)
    except ExportError as exc:
        if unsupported_as_value_error:
            raise ValueError(str(exc)) from None
        raise

    try:
        fig.savefig(
            path,
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            **kwargs,
        )
    except OSError as exc:
        raise ExportError(f"Failed to write '{path}': {exc}") from exc
    finally:
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)


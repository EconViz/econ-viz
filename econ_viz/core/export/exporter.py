"""Shared figure-export helpers."""

from __future__ import annotations

from matplotlib.figure import Figure as MplFigure

from econ_viz.core.constants.io import SAVEFIG_BBOX_INCHES
from econ_viz.core.errors.exceptions import ExportError
from econ_viz.enums import ExportFormat


def save_figure(
    fig: MplFigure,
    *,
    path: str,
    dpi: int,
    close: bool = False,
    unsupported_as_value_error: bool = False,
    transparent: bool = True,
    **kwargs,
) -> None:
    """Save a matplotlib figure with consistent format validation and errors."""
    try:
        fmt = ExportFormat.from_path(path)
    except ExportError as exc:
        if unsupported_as_value_error:
            raise ValueError(str(exc)) from None
        raise

    tikz_scale = kwargs.pop("tikz_scale", None)
    tikz_standalone = kwargs.pop("tikz_standalone", True)

    try:
        if fmt is ExportFormat.TEX:
            from econ_viz.core.export.backend_tikz import save_tikz

            save_tikz(
                fig,
                path,
                scale=tikz_scale,
                standalone=bool(tikz_standalone),
            )
            return

        fig.savefig(
            path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=SAVEFIG_BBOX_INCHES,
            **kwargs,
        )
    except OSError as exc:
        raise ExportError(f"Failed to write '{path}': {exc}") from exc
    finally:
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)

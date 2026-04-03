"""Multi-panel figure assembly for economic diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.gridspec import GridSpec

from .base import Canvas
from ..enums import Layout
from ..io import save_figure
from ..themes import default as _default_theme
from ..themes.theme import Theme


@dataclass(frozen=True)
class _PanelSpec:
    """GridSpec placement for one panel."""

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


_LAYOUT_SPECS: Final[dict[Layout, tuple[tuple[int, int], list[_PanelSpec]]]] = {
    Layout.SINGLE: ((1, 1), [_PanelSpec(0, 0)]),
    Layout.STACKED: ((2, 1), [_PanelSpec(0, 0), _PanelSpec(1, 0)]),
    Layout.SIDE_BY_SIDE: ((1, 2), [_PanelSpec(0, 0), _PanelSpec(0, 1)]),
    Layout.TOP_TWO_BOTTOM_ONE: ((2, 2), [_PanelSpec(0, 0), _PanelSpec(0, 1), _PanelSpec(1, 0, colspan=2)]),
    Layout.TOP_ONE_BOTTOM_TWO: ((2, 2), [_PanelSpec(0, 0, colspan=2), _PanelSpec(1, 0), _PanelSpec(1, 1)]),
    Layout.GRID_2X2: ((2, 2), [_PanelSpec(0, 0), _PanelSpec(0, 1), _PanelSpec(1, 0), _PanelSpec(1, 1)]),
    Layout.GRID_3X3: (
        (3, 3),
        [_PanelSpec(r, c) for r in range(3) for c in range(3)],
    ),
}


class Figure:
    """A fixed-layout collection of canvases sharing one matplotlib figure.

    Parameters
    ----------
    layout : Layout
        Named panel arrangement.
    x_max, y_max : float
        Axis limits applied to every panel.
    x_label, y_label : str
        Axis labels applied to every panel.
    title : str | None
        Optional super-title for the whole figure.
    dpi : int
        Export resolution passed through to panel canvases.
    x_label_pos, y_label_pos : str
        Per-panel label placement options forwarded to :class:`Canvas`.
    theme : Theme
        Theme shared across all panels.
    shared_x, shared_y : bool
        When enabled, matplotlib links axes and inner-edge axis-tip labels are
        suppressed.
    figsize : tuple[float, float] | None
        Optional explicit figure size. Defaults to ``(6 * cols, 6 * rows)``.
    hspace, wspace : float
        GridSpec spacing parameters.
    """

    def __init__(
        self,
        layout: Layout,
        x_max: float = 10.0,
        y_max: float = 10.0,
        x_label: str = "X",
        y_label: str = "Y",
        title: str | None = None,
        dpi: int = 300,
        x_label_pos: str = "right",
        y_label_pos: str = "top",
        theme: Theme = _default_theme,
        shared_x: bool = False,
        shared_y: bool = False,
        figsize: tuple[float, float] | None = None,
        hspace: float = 0.3,
        wspace: float = 0.25,
    ):
        """Create a multi-panel figure composed of injected :class:`Canvas` instances."""
        self.layout = layout
        self.shared_x = shared_x
        self.shared_y = shared_y
        shape, specs = _LAYOUT_SPECS[layout]
        rows, cols = shape
        width = 6.0 * cols
        height = 6.0 * rows
        self.fig: MplFigure = plt.figure(figsize=figsize or (width, height))
        self.fig.patch.set_alpha(0.0)
        if title:
            self.fig.suptitle(title, color=theme.label_color)

        gs = GridSpec(rows, cols, figure=self.fig, hspace=hspace, wspace=wspace)
        self.canvases: list[Canvas] = []
        self._grid_lookup: dict[tuple[int, int], Canvas] = {}
        anchor_ax: Axes | None = None

        for spec in specs:
            sharex_ax = anchor_ax if shared_x else None
            sharey_ax = anchor_ax if shared_y else None
            ax = self.fig.add_subplot(
                gs[spec.row:spec.row + spec.rowspan, spec.col:spec.col + spec.colspan],
                sharex=sharex_ax,
                sharey=sharey_ax,
            )
            if anchor_ax is None:
                anchor_ax = ax
            canvas = Canvas(
                x_max=x_max,
                y_max=y_max,
                x_label=x_label,
                y_label=y_label,
                title=None,
                dpi=dpi,
                x_label_pos=x_label_pos,
                y_label_pos=y_label_pos,
                theme=theme,
                fig=self.fig,
                ax=ax,
            )
            self.canvases.append(canvas)
            self._grid_lookup[(spec.row, spec.col)] = canvas

        self._apply_shared_axes(shape, specs)

    def _apply_shared_axes(self, shape: tuple[int, int], specs: list[_PanelSpec]) -> None:
        """Hide duplicated axis-tip labels on inner edges for shared-axis layouts."""
        rows, cols = shape
        for spec, canvas in zip(specs, self.canvases):
            show_x = not self.shared_x or (spec.row + spec.rowspan == rows)
            show_y = not self.shared_y or spec.col == 0
            canvas.set_axis_visibility(show_x_label=show_x, show_y_label=show_y)

    def __getitem__(self, idx: int | tuple[int, int]) -> Canvas:
        """Return a panel canvas by flat index or ``(row, col)`` location."""
        if isinstance(idx, tuple):
            return self._grid_lookup[idx]
        return self.canvases[idx]

    def __len__(self) -> int:
        """Return the number of panels in the layout."""
        return len(self.canvases)

    def save(self, path: str, **kwargs) -> None:
        """Save the full multi-panel figure to ``.png``, ``.svg``, or ``.pdf``.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        ExportError
            If writing to disk fails.
        """
        save_figure(
            self.fig,
            path=path,
            dpi=self.canvases[0].dpi if self.canvases else 300,
            close=True,
            unsupported_as_value_error=True,
            **kwargs,
        )

    def show(self) -> None:
        """Display the multi-panel figure interactively."""
        self.fig.show()

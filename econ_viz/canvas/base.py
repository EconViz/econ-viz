"""Canvas — the central plotting surface for economic diagrams.

A :class:`Canvas` instance manages a single matplotlib figure styled in the
convention of microeconomic textbook diagrams: first-quadrant axes with
LaTeX-rendered labels at the axis tips, origin marker, arrow terminators,
and no numeric tick labels.

Drawing is delegated to component classes in :mod:`econ_viz.components`;
the canvas itself is a thin orchestration layer that resolves theme
defaults and forwards calls.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from typing import Callable

from ..utils.logging import get_logger
from ..themes import default as _default_theme
from ..themes.theme import Theme
from ..canvas.primitives import annotate_math, plot_point
from ..canvas.renderers import render_budget, render_equilibrium, render_path, render_utility
from ..io import save_figure

logger = get_logger(__name__)

_MAX_DPI = 1200
_DEFAULT_DPI = 300
_MATH_CHARS = {"^", "_", "{", "}", "\\"}


def _label_math(text: str) -> str:
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        return text
    return rf"${text}$"


def _math_wrap(text: str) -> str:
    """Wrap substrings containing LaTeX math characters in ``$...$``.

    Segments already enclosed in ``$...$`` are left untouched.  Plain-text
    segments that contain any of ``^ _ { } \\`` are automatically wrapped so
    that matplotlib renders them via its mathtext engine.
    """
    import re
    parts = re.split(r"(\$[^$]+\$)", text)
    out = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
        elif any(c in part for c in _MATH_CHARS):
            out.append(f"${part}$")
        else:
            out.append(part)
    return "".join(out)


def _smooth_xy(xs: list[float], ys: list[float], n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return a parametric spline through ``(xs, ys)`` or the raw data as fallback."""
    if len(xs) < 3 or len(ys) < 3:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    try:
        from scipy.interpolate import make_interp_spline

        points = np.column_stack((xs, ys))
        diffs = np.diff(points, axis=0)
        chord = np.sqrt((diffs ** 2).sum(axis=1))
        t = np.concatenate(([0.0], np.cumsum(chord)))
        if np.isclose(t[-1], 0.0):
            return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        k = min(3, len(xs) - 1)
        t_new = np.linspace(0.0, t[-1], max(n_samples, len(xs)))
        spline_x = make_interp_spline(t, np.asarray(xs, dtype=float), k=k)
        spline_y = make_interp_spline(t, np.asarray(ys, dtype=float), k=k)
        return spline_x(t_new), spline_y(t_new)
    except Exception:  # pragma: no cover - fallback is intentionally conservative
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _extend_curve_endpoints(
    xs: np.ndarray,
    ys: np.ndarray,
    extension_frac: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a curve slightly past its first and last points along endpoint tangents."""
    if len(xs) < 2 or len(ys) < 2 or extension_frac <= 0.0:
        return xs, ys

    start_vec = np.array([xs[1] - xs[0], ys[1] - ys[0]], dtype=float)
    end_vec = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2]], dtype=float)
    total_scale = max(
        float(np.max(xs) - np.min(xs)),
        float(np.max(ys) - np.min(ys)),
        1.0,
    )

    def _extended_point(point_x: float, point_y: float, tangent: np.ndarray, sign: float) -> tuple[float, float]:
        norm = float(np.linalg.norm(tangent))
        if np.isclose(norm, 0.0):
            return point_x, point_y
        step = tangent / norm * (total_scale * extension_frac * sign)
        return point_x + step[0], point_y + step[1]

    start_x, start_y = _extended_point(xs[0], ys[0], start_vec, -1.0)
    end_x, end_y = _extended_point(xs[-1], ys[-1], end_vec, 1.0)
    return (
        np.concatenate(([start_x], xs, [end_x])),
        np.concatenate(([start_y], ys, [end_y])),
    )


class Canvas:
    """First-quadrant plotting surface for economic visualizations.

    Parameters
    ----------
    x_max : float
        Upper bound of the horizontal axis.
    y_max : float
        Upper bound of the vertical axis.
    x_label : str
        Label at the tip of the horizontal axis (rendered in LaTeX math mode).
    y_label : str
        Label at the tip of the vertical axis (rendered in LaTeX math mode).
    title : str or None
        Optional figure title.
    dpi : int
        Resolution for raster export. Clamped to ``[1, 1200]``. Default 300.
    x_label_pos : str
        Position of the x-axis label: ``"right"`` (default, at axis tip) or
        ``"bottom"`` (centred below the axis).
    y_label_pos : str
        Position of the y-axis label: ``"top"`` (default, at axis tip) or
        ``"left"`` (centred to the left of the axis).
    theme : Theme
        Colour and style theme. Defaults to the built-in ``default`` theme.
    """

    def __init__(
        self,
        x_max: float = 10.0,
        y_max: float = 10.0,
        x_label: str = "X",
        y_label: str = "Y",
        title: str | None = None,
        dpi: int = _DEFAULT_DPI,
        x_label_pos: str = "right",
        y_label_pos: str = "top",
        theme: Theme = _default_theme,
        fig=None,
        ax=None,
    ):
        self.x_max = x_max
        self.y_max = y_max
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.dpi = max(1, min(dpi, _MAX_DPI))
        self.x_label_pos = x_label_pos
        self.y_label_pos = y_label_pos
        self.theme = theme

        self._owns_figure = fig is None or ax is None
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        else:
            self.fig, self.ax = fig, ax
        self._legend_handles: list = []
        self._apply_base_style()
        logger.debug("Canvas created: x_max=%s, y_max=%s, dpi=%s, theme=%s",
                     x_max, y_max, self.dpi, theme.name)

    # ------------------------------------------------------------------
    # Base styling
    # ------------------------------------------------------------------

    def _apply_base_style(self) -> None:
        """Configure axes to match textbook economic diagram conventions."""
        t = self.theme
        self.ax.set_xlim(0, self.x_max)
        self.ax.set_ylim(0, self.y_max)

        # Turn off tick labels
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)

        # X-axis label
        if self.x_label_pos == "right":
            self.ax.text(
                self.x_max, -self.y_max * 0.03,
                _label_math(self.x_label), ha="center", va="top",
                fontsize=14, color=t.label_color,
            )
        else:  # bottom
            self.ax.set_xlabel(_label_math(self.x_label), fontsize=14, color=t.label_color)

        # Y-axis label
        if self.y_label_pos == "top":
            self.ax.text(
                -self.x_max * 0.03, self.y_max,
                _label_math(self.y_label), ha="right", va="center",
                fontsize=14, color=t.label_color,
            )
        else:  # left
            self.ax.set_ylabel(_label_math(self.y_label), fontsize=14,
                               rotation=0, color=t.label_color)

        # Origin label
        self.ax.text(
            -self.x_max * 0.03, -self.y_max * 0.03,
            r"$0$", ha="right", va="top", fontsize=12, color=t.label_color,
        )

        if self.title:
            self.ax.set_title(_math_wrap(self.title), color=t.label_color)

        # Spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_color(t.axis_color)
        self.ax.spines["left"].set_color(t.axis_color)

        # Arrow terminators at axis tips
        self.ax.plot(self.x_max, 0, ">", color=t.axis_color, markersize=7, clip_on=False)
        self.ax.plot(0, self.y_max, "^", color=t.axis_color, markersize=7, clip_on=False)

        # Transparent background
        self.fig.patch.set_alpha(0.0)
        self.ax.patch.set_alpha(0.0)

    def set_axis_visibility(self, *, show_x_label: bool = True, show_y_label: bool = True) -> Canvas:
        """Toggle canvas axis-tip labels and origin marker for shared layouts."""
        if not show_x_label:
            for text in list(self.ax.texts):
                x, y = text.get_position()
                if abs(y + self.y_max * 0.03) < max(self.y_max, 1.0) * 1e-9:
                    text.set_visible(False)
        if not show_y_label:
            for text in list(self.ax.texts):
                x, y = text.get_position()
                if abs(x + self.x_max * 0.03) < max(self.x_max, 1.0) * 1e-9:
                    text.set_visible(False)
        return self

    # ------------------------------------------------------------------
    # Layer composition
    # ------------------------------------------------------------------

    def add_utility(
        self,
        func: Callable,
        levels: int | list = 3,
        color: str | None = None,
        linewidth: float | None = None,
        show_rays: bool = False,
        show_kinks: bool = False,
        kink_radius: float = 1.0,
        show_bliss: bool = True,
        label: str | None = None,
        show_ic_labels: bool = False,
        ic_label_fmt: str = "{:.2g}",
        **kwargs,
    ) -> Canvas:
        """Add indifference curves for a given utility function.

        Colour and line-width fall back to the active theme when not
        specified explicitly.

        Parameters
        ----------
        func : UtilityFunction
            A utility model conforming to the :class:`UtilityFunction` protocol.
        levels : int or list
            Number of contour levels (automatically spaced by percentile) or an
            explicit list of utility values at which to draw contours.
        color : str or None
            Curve colour. *None* → ``theme.ic_color``.
        linewidth : float or None
            Stroke width. *None* → ``theme.ic_linewidth``.
        show_rays : bool
            If ``True``, draw dashed kink-locus rays (KINKED types only).
        show_kinks : bool
            If ``True`` and ``func.utility_type is KINKED``, draw circular
            markers at kink points on each contour level.
        kink_radius : float
            Marker size factor for kink dots.
        show_bliss : bool
            If ``True`` (default) and *func* has ``bliss_x`` / ``bliss_y``
            attributes (i.e. a :class:`~econ_viz.models.Satiation` model),
            draw a star marker at the bliss point.
        label : str or None
            Legend label for this indifference curve family.  Appears in the
            legend when :meth:`show_legend` is called.
        show_ic_labels : bool
            If ``True``, place a small text label at the right end of each
            indifference curve showing its utility level.
        ic_label_fmt : str
            Python format string for the level value (default ``"{:.2g}"``).
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.contour`.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        t = self.theme
        ic = render_utility(
            self.ax,
            func=func,
            levels=levels,
            color=color or t.ic_color,
            linewidth=linewidth if linewidth is not None else t.ic_linewidth,
            show_rays=show_rays,
            ray_color=t.ray_color,
            ray_linewidth=t.ray_linewidth,
            show_kinks=show_kinks,
            kink_color=t.kink_color,
            kink_radius=kink_radius,
            label=label,
            show_ic_labels=show_ic_labels,
            ic_label_fmt=ic_label_fmt,
            show_bliss=show_bliss,
            x_max=self.x_max,
            y_max=self.y_max,
            **kwargs,
        )
        if ic._proxy is not None:
            self._legend_handles.append(ic._proxy)
        return self

    def add_budget(
        self,
        px: float,
        py: float,
        income: float,
        color: str | None = None,
        linewidth: float | None = None,
        linestyle: str = "-",
        label: str | None = None,
        fill: bool = False,
        fill_alpha: float | None = None,
    ) -> Canvas:
        """Add a linear budget constraint px*x + py*y = income.

        Colour, line-width, and fill opacity fall back to the active
        theme when not specified explicitly.

        Parameters
        ----------
        px, py : float
            Prices. Must be positive.
        income : float
            Total budget. Must be positive.
        color : str or None
            Line colour. *None* → ``theme.budget_color``.
        linewidth : float or None
            Stroke width. *None* → ``theme.budget_linewidth``.
        linestyle : str
            Matplotlib line-style string.
        label : str or None
            Optional legend label rendered in LaTeX math mode.
        fill : bool
            If ``True``, shade the feasible set below the budget line.
        fill_alpha : float or None
            Opacity of the shading. *None* → ``theme.budget_fill_alpha``.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        t = self.theme
        render_budget(
            self.ax,
            px=px,
            py=py,
            income=income,
            color=color or t.budget_color,
            linewidth=linewidth if linewidth is not None else t.budget_linewidth,
            linestyle=linestyle,
            label=label,
            fill=fill,
            fill_alpha=fill_alpha if fill_alpha is not None else t.budget_fill_alpha,
        )
        return self

    def add_equilibrium(
        self,
        eq,
        color: str | None = None,
        markersize: float | None = None,
        label: str | None = "x^*",
        drop_dashes: bool = True,
        show_ray: bool = False,
    ) -> Canvas:
        """Annotate a pre-solved equilibrium on the canvas.

        Call :func:`~econ_viz.optimizer.solve` first, then pass the
        resulting :class:`~econ_viz.optimizer.Equilibrium` here.

        Parameters
        ----------
        eq : Equilibrium
            A solved equilibrium bundle.
        color : str or None
            Marker / drop-line colour. *None* → ``theme.eq_color``.
        markersize : float or None
            Dot size. *None* → ``theme.eq_markersize``.
        label : str or None
            LaTeX label placed next to the dot.
        drop_dashes : bool
            Draw dashed perpendicular lines from the optimum to both axes.
        show_ray : bool
            Draw the expansion-path ray from the origin through the optimum.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        t = self.theme
        render_equilibrium(
            self.ax,
            eq=eq,
            color=color or t.eq_color,
            markersize=markersize if markersize is not None else t.eq_markersize,
            label=label,
            drop_dashes=drop_dashes,
            show_ray=show_ray,
            ray_color=t.ray_color,
            ray_linewidth=t.ray_linewidth,
            x_max=self.x_max,
            y_max=self.y_max,
        )
        return self

    def add_ray(
        self,
        slope: float,
        color: str | None = None,
        linewidth: float | None = None,
    ) -> Canvas:
        """Add a dashed ray emanating from the origin.

        Parameters
        ----------
        slope : float
            Rise-over-run slope of the ray (dy / dx).
        color : str or None
            *None* → ``theme.ray_color``.
        linewidth : float or None
            *None* → ``theme.ray_linewidth``.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        from ..components import draw_ray

        t = self.theme
        draw_ray(
            self.ax, slope, self.x_max, self.y_max,
            color=color or t.ray_color,
            linewidth=linewidth if linewidth is not None else t.ray_linewidth,
        )
        return self

    def add_point(
        self,
        x: float,
        y: float,
        label: str | None = None,
        color: str | None = None,
        markersize: float = 6.0,
        offset: tuple[float, float] = (5, 5),
    ) -> Canvas:
        """Plot a labelled point on the canvas.

        Parameters
        ----------
        x, y : float
            Coordinates of the point.
        label : str or None
            Text label (rendered in LaTeX math mode if provided).
        color : str or None
            Marker and label colour. *None* → ``theme.eq_color``.
        markersize : float
            Size of the dot.
        offset : tuple[float, float]
            ``(dx, dy)`` text offset in points from the marker centre.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        c = color or self.theme.eq_color
        plot_point(
            self.ax,
            x=x,
            y=y,
            color=c,
            markersize=markersize,
            marker="o",
            linestyle="None",
            zorder=6,
            clip_on=False,
        )
        if label:
            annotate_math(
                self.ax,
                x=x,
                y=y,
                text=label,
                color=c,
                offset=offset,
                fontsize=12,
                zorder=7,
            )
        return self

    def add_path(
        self,
        path,
        color: str | None = None,
        linewidth: float | None = None,
        label: str | None = None,
        show_points: bool | None = None,
        show_budgets: bool | None = None,
        show_curves: bool = False,
        show_equilibria: bool = False,
        invert_axes: bool = False,
        smooth_curve: bool | None = None,
    ) -> Canvas:
        """Draw a PCC/ICC-style path through a sequence of equilibria."""
        c = color or self.theme.path_color
        lw = linewidth if linewidth is not None else self.theme.path_linewidth
        show_points = path.default_show_points if show_points is None else show_points
        show_budgets = path.default_show_budgets if show_budgets is None else show_budgets
        smooth_curve = path.default_smooth_curve if smooth_curve is None else smooth_curve
        render_path(
            canvas=self,
            path=path,
            color=c,
            linewidth=lw,
            label=label,
            show_points=show_points,
            show_budgets=show_budgets,
            show_curves=show_curves,
            show_equilibria=show_equilibria,
            invert_axes=invert_axes,
            smooth_curve=smooth_curve,
            smooth_fn=_smooth_xy,
            extend_fn=_extend_curve_endpoints,
        )
        return self

    def show_legend(self, **kwargs) -> Canvas:
        """Render a legend for all labelled layers.

        Collects proxy artists registered by :meth:`add_utility`,
        :meth:`add_budget` (when a *label* is supplied), and any future
        labelled layers, then delegates to :meth:`matplotlib.axes.Axes.legend`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.legend`.  Common options:
            ``loc``, ``frameon``, ``fontsize``.

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        budget_handles, budget_labels = self.ax.get_legend_handles_labels()
        all_handles = self._legend_handles + budget_handles
        if all_handles:
            all_labels = [h.get_label() for h in self._legend_handles] + budget_labels
            kwargs.setdefault("frameon", False)
            kwargs.setdefault("fontsize", 11)
            self.ax.legend(handles=all_handles, labels=all_labels, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display the figure in an interactive matplotlib window."""
        self.fig.show()

    def save(self, path: str, **kwargs) -> None:
        """Export the figure to disk and release matplotlib resources.

        The output format is inferred from the file extension via
        :class:`~econ_viz.enums.ExportFormat`.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"plot.png"``, ``"plot.svg"``).
        **kwargs
            Forwarded to the underlying save function.
        """
        logger.info("Exporting figure to %s (dpi=%s)", path, self.dpi)
        save_figure(
            self.fig,
            path=path,
            dpi=self.dpi,
            close=self._owns_figure,
            **kwargs,
        )

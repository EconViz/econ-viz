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
from matplotlib.patches import FancyArrowPatch

from collections.abc import Sequence
from typing import Callable

from ..constants.canvas import (
    ARROW_HEAD_ONLY_FRAC,
    ARROW_WEDGE_FRAC,
    DEFAULT_DPI,
    ENDPOINT_EXTENSION_FRAC,
    MATH_CHARS,
    MAX_DPI,
    MIN_DPI,
    SMOOTH_SAMPLES,
)
from ..utils.logging import get_logger
from ..themes import default as _default_theme
from ..themes.theme import Theme
from ..enums import ArrowStyle, LabelPosition, LineStyle
from ..canvas.fonts import FontApplier, resolve_font, resolve_math_font
from ..canvas.effect import Effect
from ..canvas.stroke import styled
from ..themes.marker import Marker
from ..themes.stroke import Stroke
from ..canvas.primitives import annotate_math, plot_point
from ..canvas.renderers import (
    render_budget,
    render_decomposition,
    render_equilibrium,
    render_path,
    render_utility,
)
from ..io import save_figure

logger = get_logger(__name__)

_X_LABEL_POSITIONS = {
    LabelPosition.TOP: ((0, 8), "center", "bottom"),
    LabelPosition.RIGHT: ((8, 0), "left", "center"),
    LabelPosition.BOTTOM: ((0, -8), "center", "top"),
}
_Y_LABEL_POSITIONS = {
    LabelPosition.LEFT: ((-8, 0), "right", "center"),
    LabelPosition.TOP: ((0, 8), "center", "bottom"),
    LabelPosition.RIGHT: ((8, 0), "left", "center"),
}


def _label_position(value: LabelPosition | str, *, axis: str) -> LabelPosition:
    """Normalize and validate one axis-label position."""
    try:
        position = LabelPosition(value)
    except ValueError:
        valid = _X_LABEL_POSITIONS if axis == "x" else _Y_LABEL_POSITIONS
        choices = ", ".join(item.value for item in valid)
        raise ValueError(f"invalid {axis}-axis label position {value!r}; choose: {choices}") from None

    valid = _X_LABEL_POSITIONS if axis == "x" else _Y_LABEL_POSITIONS
    if position not in valid:
        choices = ", ".join(item.value for item in valid)
        raise ValueError(f"invalid {axis}-axis label position {value!r}; choose: {choices}")
    return position


def _line_style(value: LineStyle | str) -> LineStyle:
    """Normalize one axis line style."""
    try:
        return LineStyle(value)
    except ValueError:
        choices = ", ".join(item.value for item in LineStyle)
        raise ValueError(f"invalid line style {value!r}; choose: {choices}") from None


def _axis_stroke(theme, line_style, arrow_style, shared: Stroke | None, own: Stroke | None) -> Stroke:
    """Resolve one axis's stroke; the colour falls back to ``theme.axis_color``."""
    stroke = Stroke(
        style=_line_style(line_style) if line_style is not None else None,
        arrow=_arrow_style(arrow_style) if arrow_style is not None else None,
    ).merged_over(theme.axis_stroke)
    for override in (shared, own):
        if override is not None:
            stroke = override.merged_over(stroke)
    return stroke.merged_over(Stroke(color=theme.axis_color))


def _arrow_style(value: ArrowStyle | str) -> ArrowStyle:
    """Normalize one axis arrow style."""
    try:
        return ArrowStyle(value)
    except ValueError:
        choices = ", ".join(item.value for item in ArrowStyle)
        raise ValueError(f"invalid arrow style {value!r}; choose: {choices}") from None


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
        elif any(c in part for c in MATH_CHARS):
            out.append(f"${part}$")
        else:
            out.append(part)
    return "".join(out)


def _smooth_xy(xs: list[float], ys: list[float], n_samples: int = SMOOTH_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
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
    extension_frac: float = ENDPOINT_EXTENSION_FRAC,
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
    x_label_pos : LabelPosition or str
        Position around the x-axis arrowhead: top, right, or bottom.
    y_label_pos : LabelPosition or str
        Position around the y-axis arrowhead: left, top, or right.
    theme : Theme
        Colour and style theme. Defaults to the built-in ``default`` theme.
    x_arrow_style, y_arrow_style : ArrowStyle or str
        Independently configurable arrowhead styles for each axis.
    font : str or sequence of str, optional
        Font family for every text element on this canvas, or a fallback list.
        Generic families (``"serif"``, ``"sans-serif"``, ``"monospace"``) are
        accepted. ``None`` keeps Matplotlib's default. Global rcParams are not
        modified.
    math_font : str, optional
        Matplotlib math font set for math text such as axis labels:
        ``"dejavusans"``, ``"dejavuserif"``, ``"cm"``, ``"stix"``, or
        ``"stixsans"``. ``None`` keeps Matplotlib's default.
    x_line_style, y_line_style : LineStyle or str
        Line style of each axis: solid, dashed, dotted, or dashdot.
    axis_stroke : Stroke, optional
        Width, line style, colour, and arrowhead for both axes
        (default ``theme.axis_stroke``).
    x_axis_stroke, y_axis_stroke : Stroke, optional
        Per-axis overrides on top of *axis_stroke*.
    """

    def __init__(
        self,
        x_max: float = 10.0,
        y_max: float = 10.0,
        x_label: str = "X",
        y_label: str = "Y",
        title: str | None = None,
        dpi: int = DEFAULT_DPI,
        x_label_pos: LabelPosition | str = LabelPosition.RIGHT,
        y_label_pos: LabelPosition | str = LabelPosition.TOP,
        theme: Theme = _default_theme,
        fig=None,
        ax=None,
        x_arrow_style: ArrowStyle | str | None = None,
        y_arrow_style: ArrowStyle | str | None = None,
        font: str | Sequence[str] | None = None,
        math_font: str | None = None,
        x_line_style: LineStyle | str | None = None,
        y_line_style: LineStyle | str | None = None,
        axis_stroke: Stroke | None = None,
        x_axis_stroke: Stroke | None = None,
        y_axis_stroke: Stroke | None = None,
    ):
        self.x_max = x_max
        self.y_max = y_max
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.dpi = max(MIN_DPI, min(dpi, MAX_DPI))
        self.x_label_pos = _label_position(x_label_pos, axis="x")
        self.y_label_pos = _label_position(y_label_pos, axis="y")
        self.theme = theme
        # Precedence: per-axis stroke > shared stroke > x/y_*_style arguments > theme.axis_stroke.
        self.x_axis_stroke = _axis_stroke(theme, x_line_style, x_arrow_style, axis_stroke, x_axis_stroke)
        self.y_axis_stroke = _axis_stroke(theme, y_line_style, y_arrow_style, axis_stroke, y_axis_stroke)
        self.x_line_style = self.x_axis_stroke.style
        self.y_line_style = self.y_axis_stroke.style
        self.x_arrow_style = self.x_axis_stroke.arrow
        self.y_arrow_style = self.y_axis_stroke.arrow
        self.font = resolve_font(font)
        self.math_font = resolve_math_font(math_font)

        self._owns_figure = fig is None or ax is None
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        else:
            self.fig, self.ax = fig, ax
        if (self.font or self.math_font) and self._owns_figure:
            self.fig.add_artist(FontApplier(self.font, self.math_font))
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

        x_offset, x_ha, x_va = _X_LABEL_POSITIONS[self.x_label_pos]
        x_label_text = self.ax.annotate(
            _label_math(self.x_label),
            xy=(self.x_max, 0),
            xytext=x_offset,
            textcoords="offset points",
            ha=x_ha,
            va=x_va,
            fontsize=14,
            color=t.label_color,
            clip_on=False,
        )
        x_label_text._ev_axis_label = "x"

        y_offset, y_ha, y_va = _Y_LABEL_POSITIONS[self.y_label_pos]
        y_label_text = self.ax.annotate(
            _label_math(self.y_label),
            xy=(0, self.y_max),
            xytext=y_offset,
            textcoords="offset points",
            ha=y_ha,
            va=y_va,
            fontsize=14,
            color=t.label_color,
            clip_on=False,
        )
        y_label_text._ev_axis_label = "y"

        # Origin label
        self.ax.text(
            -self.x_max * 0.03, -self.y_max * 0.03,
            r"$0$", ha="right", va="top", fontsize=12, color=t.label_color,
        )

        if self.title:
            self.ax.set_title(_math_wrap(self.title), color=t.label_color, pad=18)

        # Spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        for spine, stroke in (("bottom", self.x_axis_stroke), ("left", self.y_axis_stroke)):
            self.ax.spines[spine].set_color(stroke.color)
            self.ax.spines[spine].set_linewidth(stroke.width)
            self.ax.spines[spine].set_linestyle(stroke.style.value)

        # Arrow terminators at axis tips
        def arrow_frac(style: ArrowStyle) -> float:
            return ARROW_WEDGE_FRAC if style is ArrowStyle.WEDGE else ARROW_HEAD_ONLY_FRAC

        for axis, stroke, tip in (
            ("x", self.x_axis_stroke, (self.x_max, 0)),
            ("y", self.y_axis_stroke, (0, self.y_max)),
        ):
            if stroke.arrow is None:
                continue
            frac = arrow_frac(stroke.arrow)
            start = (self.x_max * (1 - frac), 0) if axis == "x" else (0, self.y_max * (1 - frac))
            arrow = FancyArrowPatch(
                start,
                tip,
                arrowstyle=stroke.arrow.value,
                mutation_scale=12,
                linewidth=stroke.width,
                color=stroke.color,
                shrinkA=0,
                shrinkB=0,
                clip_on=False,
            )
            arrow._ev_axis_arrow = axis
            arrow._ev_arrow_style = stroke.arrow
            self.ax.add_patch(arrow)

        # Transparent background
        self.fig.patch.set_alpha(0.0)
        self.ax.patch.set_alpha(0.0)

    def set_axis_visibility(self, *, show_x_label: bool = True, show_y_label: bool = True) -> Canvas:
        """Toggle canvas axis-tip labels and origin marker for shared layouts."""
        if not show_x_label:
            for text in list(self.ax.texts):
                if getattr(text, "_ev_axis_label", None) == "x":
                    text.set_visible(False)
        if not show_y_label:
            for text in list(self.ax.texts):
                if getattr(text, "_ev_axis_label", None) == "y":
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
        stroke: Stroke | None = None,
        ray_stroke: Stroke | None = None,
        kink_marker: Marker | None = None,
        bliss_marker: Marker | None = None,
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

        stroke : Stroke, optional
            Line style for the indifference curves (default ``theme.ic_stroke``); ``arrow`` adds an arrowhead to each curve.
        ray_stroke : Stroke, optional
            Line style for kink-locus rays (default ``theme.ray_stroke``).

        kink_marker : Marker, optional
            Colour, size, and shape of kink points (default ``theme.kink_marker``).
        bliss_marker : Marker, optional
            Colour, size, and shape of the bliss point (default ``theme.bliss_marker``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        with styled(self, {"curve": stroke, "ray": ray_stroke}, markers={"kink": kink_marker, "bliss": bliss_marker}):
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
        stroke: Stroke | None = None,
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

        stroke : Stroke, optional
            Line style for the budget line (default ``theme.budget_stroke``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        with styled(self, {"budget": stroke}):
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
        drop_stroke: Stroke | None = None,
        ray_stroke: Stroke | None = None,
        marker: Marker | None = None,
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

        drop_stroke : Stroke, optional
            Line style for the guides to both axes (default ``theme.drop_stroke``).
        ray_stroke : Stroke, optional
            Line style for the expansion-path ray (default ``theme.ray_stroke``).

        marker : Marker, optional
            Colour, size, and shape of the equilibrium point (default ``theme.eq_marker``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        with styled(self, {"drop": drop_stroke, "ray": ray_stroke}, markers={"equilibrium": marker}):
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

    def add_decomposition(
        self,
        decomposition,
        *,
        show_arrows: bool = True,
        label_effects: bool = True,
        show_x_projections: bool = False,
        point_color: str | None = None,
        point_markersize: float | None = None,
        original_budget_color: str | None = None,
        original_budget_linewidth: float | None = None,
        original_budget_linestyle: str = "-",
        compensated_budget_color: str | None = None,
        compensated_budget_linewidth: float | None = None,
        compensated_budget_linestyle: str | None = None,
        final_budget_color: str | None = None,
        final_budget_linewidth: float | None = None,
        final_budget_linestyle: str = "-.",
        substitution_color: str | None = None,
        income_color: str | None = None,
        effect_arrow_linewidth: float | None = None,
        original_budget_stroke: Stroke | None = None,
        compensated_budget_stroke: Stroke | None = None,
        final_budget_stroke: Stroke | None = None,
        substitution_stroke: Stroke | None = None,
        income_stroke: Stroke | None = None,
        projection_stroke: Stroke | None = None,
        guide_stroke: Stroke | None = None,
        range_stroke: Stroke | None = None,
        substitution: Effect | None = None,
        income: Effect | None = None,
        point_marker: Marker | None = None,
    ) -> Canvas:
        """Render a Hicks/Slutsky price-effect decomposition on this canvas.

        Parameters
        ----------
        decomposition : PriceEffectDecomposition
            Result returned by :func:`econ_viz.optimizer.decompose_price_effect`.
        show_arrows : bool
            If ``True`` draw substitution/income arrows.
        label_effects : bool
            If ``True`` annotate effect magnitudes near each arrow.
        show_x_projections : bool
            If ``True`` draw x-axis projection guides and brackets.
        original_budget_stroke : Stroke, optional
            Line style for the original budget line (default ``theme.budget_stroke``).
        compensated_budget_stroke : Stroke, optional
            Line style for the compensated budget line (default ``theme.compensated_budget_stroke``).
        final_budget_stroke : Stroke, optional
            Line style for the final budget line (default ``theme.final_budget_stroke``).
        substitution_stroke : Stroke, optional
            Line style for the substitution-effect arrow (default ``theme.substitution_stroke``).
        income_stroke : Stroke, optional
            Line style for the income-effect arrow (default ``theme.income_stroke``).
        projection_stroke : Stroke, optional
            Line style for vertical guides from A, B, C to the x-axis (default ``theme.projection_stroke``).
        guide_stroke : Stroke, optional
            Line style for guides below the x-axis (default ``theme.guide_stroke``).
        range_stroke : Stroke, optional
            Line style for effect-size arrows below the x-axis.
        substitution, income : Effect, optional
            Colour, range-arrow height, and label of each effect. The colour
            overrides *substitution_color* / *income_color*; a Stroke colour
            overrides both.
        point_marker : Marker, optional
            Colour, size, and shape of bundles A, B, C and their legend entries (default ``theme.eq_marker``).
        """
        with styled(self, {"original_budget": original_budget_stroke, "compensated_budget": compensated_budget_stroke, "final_budget": final_budget_stroke, "substitution": substitution_stroke, "income": income_stroke, "projection": projection_stroke, "guide": guide_stroke, "range": range_stroke}, markers={"bundle": point_marker}):
            if substitution is not None and substitution.color is not None:
                substitution_color = substitution.color
            if income is not None and income.color is not None:
                income_color = income.color
            t = self.theme
            render_decomposition(
                self.ax,
                decomposition=decomposition,
                point_color=point_color or t.eq_color,
                point_markersize=(
                    point_markersize if point_markersize is not None else t.eq_markersize
                ),
                original_budget_color=original_budget_color or t.budget_color,
                original_budget_linewidth=(
                    original_budget_linewidth
                    if original_budget_linewidth is not None
                    else t.budget_linewidth
                ),
                original_budget_linestyle=original_budget_linestyle,
                compensated_budget_color=(
                    compensated_budget_color or t.compensated_budget_color
                ),
                compensated_budget_linewidth=(
                    compensated_budget_linewidth
                    if compensated_budget_linewidth is not None
                    else t.compensated_budget_linewidth
                ),
                compensated_budget_linestyle=(
                    compensated_budget_linestyle
                    if compensated_budget_linestyle is not None
                    else t.compensated_budget_linestyle
                ),
                final_budget_color=final_budget_color or t.budget_color,
                final_budget_linewidth=(
                    final_budget_linewidth
                    if final_budget_linewidth is not None
                    else t.budget_linewidth
                ),
                final_budget_linestyle=final_budget_linestyle,
                show_arrows=show_arrows,
                arrows_below_axis=show_x_projections,
                substitution_color=substitution_color or t.sub_effect_color,
                income_color=income_color or t.inc_effect_color,
                effect_arrow_linewidth=(
                    effect_arrow_linewidth
                    if effect_arrow_linewidth is not None
                    else t.effect_arrow_linewidth
                ),
                show_x_projections=show_x_projections,
                substitution_effect=substitution,
                income_effect=income,
            )
            if label_effects:
                sub_dx, sub_dy = decomposition.substitution_effect
                inc_dx, inc_dy = decomposition.income_effect
                self._legend_handles.extend([
                    mlines.Line2D(
                        [],
                        [],
                        color=point_color or t.eq_color,
                        marker="o",
                        linestyle="None",
                        markersize=point_markersize if point_markersize is not None else t.eq_markersize,
                        label=rf"$A=({decomposition.A.x:.2f},{decomposition.A.y:.2f})$",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=point_color or t.eq_color,
                        marker="o",
                        linestyle="None",
                        markersize=point_markersize if point_markersize is not None else t.eq_markersize,
                        label=rf"$B=({decomposition.B.x:.2f},{decomposition.B.y:.2f})$",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=point_color or t.eq_color,
                        marker="o",
                        linestyle="None",
                        markersize=point_markersize if point_markersize is not None else t.eq_markersize,
                        label=rf"$C=({decomposition.C.x:.2f},{decomposition.C.y:.2f})$",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=substitution_color or t.sub_effect_color,
                        linestyle="--",
                        linewidth=effect_arrow_linewidth if effect_arrow_linewidth is not None else t.effect_arrow_linewidth,
                        label=rf"$Sub:\ \Delta x={sub_dx:+.2f},\ \Delta y={sub_dy:+.2f}$",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=income_color or t.inc_effect_color,
                        linestyle="--",
                        linewidth=effect_arrow_linewidth if effect_arrow_linewidth is not None else t.effect_arrow_linewidth,
                        label=rf"$Inc:\ \Delta x={inc_dx:+.2f},\ \Delta y={inc_dy:+.2f}$",
                    ),
                ])
                for handle in self._legend_handles[-5:-2]:
                    handle._ev_role = "bundle"
                self._legend_handles[-2]._ev_role = "substitution"
                self._legend_handles[-1]._ev_role = "income"
                self.show_legend(loc="upper right")
        return self

    def add_ray(
        self,
        slope: float,
        color: str | None = None,
        linewidth: float | None = None,
        stroke: Stroke | None = None,
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

        stroke : Stroke, optional
            Line style for the ray (default ``theme.ray_stroke``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        with styled(self, {"ray": stroke}):
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
        marker: Marker | None = None,
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

        marker : Marker, optional
            Colour, size, and shape of the point (default ``theme.point_marker``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        with styled(self, {}, markers={"point": marker}):
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
                role="point",
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
        stroke: Stroke | None = None,
        budget_stroke: Stroke | None = None,
        curve_stroke: Stroke | None = None,
        point_marker: Marker | None = None,
        equilibrium_marker: Marker | None = None,
    ) -> Canvas:
        """Draw a PCC/ICC-style path through a sequence of equilibria.
        stroke : Stroke, optional
            Line style for the path line (default ``theme.path_stroke``).
        budget_stroke : Stroke, optional
            Line style for budget lines drawn with ``show_budgets``.
        curve_stroke : Stroke, optional
            Line style for indifference curves drawn with ``show_curves``.
        point_marker : Marker, optional
            Colour, size, and shape of points drawn with ``show_points`` (default ``theme.path_marker``).
        equilibrium_marker : Marker, optional
            Colour, size, and shape of points drawn with ``show_equilibria``.
        """
        with styled(self, {"path": stroke, "budget": budget_stroke, "curve": curve_stroke}, markers={"path_point": point_marker, "equilibrium": equilibrium_marker}):
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
            Destination file path (e.g. ``"plot.png"``, ``"plot.svg"``,
            ``"plot.tex"``).
        **kwargs
            Forwarded to the underlying save function. TikZ exports accept
            ``tikz_scale`` and ``tikz_standalone``.
        """
        logger.info("Exporting figure to %s (dpi=%s)", path, self.dpi)
        save_figure(
            self.fig,
            path=path,
            dpi=self.dpi,
            close=self._owns_figure,
            **kwargs,
        )

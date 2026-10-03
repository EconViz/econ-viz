"""Layer composition: indifference curves, budgets, equilibria, rays, points and paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from utility_viz.core.canvas._curves import _extend_curve_endpoints, _smooth_xy
from utility_viz.core.canvas._state import CanvasState
from utility_viz.core.canvas.renderers import render_budget, render_equilibrium, render_path, render_utility
from utility_viz.core.rendering.primitives import annotate_math, plot_point
from utility_viz.core.rendering.stroke import styled
from utility_viz.core.styles.fill import Fill
from utility_viz.core.styles.label import Label, split_label
from utility_viz.core.styles.marker import Marker
from utility_viz.core.styles.stroke import Stroke
from utility_viz.enums import LabelPosition, LineStyle

if TYPE_CHECKING:
    from typing_extensions import Self


class LayerMixin(CanvasState):
    """The ``add_*`` methods of :class:`~utility_viz.core.canvas.base.Canvas` that draw one layer each."""

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
        ic_label: Label | None = None,
        bliss_label: str | Label | None = None,
        highlight_level: float | None = None,
        secondary_stroke: Stroke | None = None,
        label_style: str = "numeric",
        **kwargs,
    ) -> Self:
        """Add indifference curves for a given utility function.

        Prefer *stroke* / *ray_stroke* for line styling; *color* and *linewidth*
        are shorthand that a :class:`Stroke` supersedes.

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
            attributes (i.e. a :class:`~utility_viz.models.utility.Satiation` model),
            draw a star marker at the bliss point.
        label : str or None
            Legend label for this indifference curve family.  Appears in the
            legend when :meth:`show_legend` is called.
        show_ic_labels : bool
            If ``True``, place a small text label at the right end of each
            indifference curve showing its utility level.
        ic_label_fmt : str
            Python format string for the level value (default ``"{:.2g}"``).
        highlight_level : float, optional
            Utility level to draw at full weight; every other level is drawn
            subdued using ``secondary_stroke`` (default ``theme.secondary_ic_stroke``).
            ``None`` (default) draws every level with the same weight.
        label_style : str
            ``"numeric"`` (default) labels each curve with its formatted
            utility value; ``"ordinal"`` labels them ``u_1, u_2, ...`` in
            ascending order. Only takes effect when curve labels are shown.
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.contour`.

        stroke : Stroke, optional
            Line style for the indifference curves (default ``theme.ic_stroke``);
            ``arrow`` adds an arrowhead to each curve.
        ray_stroke : Stroke, optional
            Line style for kink-locus rays (default ``theme.ray_stroke``).
        secondary_stroke : Stroke, optional
            Line style for non-highlighted levels when *highlight_level* is
            set (default ``theme.secondary_ic_stroke``).

        kink_marker : Marker, optional
            Colour, size, and shape of kink points (default ``theme.kink_marker``).
        bliss_marker : Marker, optional
            Colour, size, and shape of the bliss point (default ``theme.bliss_marker``).

        ic_label : Label, optional
            Utility-level labels at the right end of each curve (default
            ``theme.ic_label``); passing one turns them on. Its *text* is the
            format string, like *ic_label_fmt*.
        bliss_label : str or Label, optional
            Text and placement of the bliss-point label (default ``theme.bliss_label``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        t = self.theme
        ic_fmt, ic_style = split_label(ic_label, t.ic_label, ic_label_fmt)
        bliss_text, bliss_style = split_label(bliss_label, t.bliss_label, t.bliss_label.text)
        labels = {
            "ic_label": ic_style,
            "secondary_ic_label": ic_style,
            "bliss_label": bliss_style,
        }
        markers = {"kink": kink_marker, "bliss": bliss_marker}
        sec = (secondary_stroke or Stroke()).merged_over(t.secondary_ic_stroke)
        with styled(
            self,
            {"curve": stroke, "ray": ray_stroke, "secondary_curve": secondary_stroke},
            markers=markers,
            labels=labels,
        ):
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
                show_ic_labels=show_ic_labels or ic_label is not None,
                ic_label_fmt=ic_fmt or ic_label_fmt,
                show_bliss=show_bliss,
                bliss_text=bliss_text or t.bliss_label.text or "x^*",
                bliss_markersize=t.bliss_marker.size if t.bliss_marker.size is not None else 12.0,
                subsistence_color=t.subsistence_color,
                subsistence_linewidth=t.subsistence_linewidth,
                x_max=self.x_max,
                y_max=self.y_max,
                highlight_level=highlight_level,
                secondary_color=sec.color or t.secondary_ic_color or t.ic_color,
                secondary_linewidth=sec.width if sec.width is not None else t.secondary_ic_linewidth,
                secondary_opacity=sec.opacity if sec.opacity is not None else t.secondary_ic_opacity,
                label_style=label_style,
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
        linestyle: str | None = None,
        label: str | None = None,
        fill: bool | Fill = False,
        fill_alpha: float | None = None,
        stroke: Stroke | None = None,
    ) -> Self:
        """Add a linear budget constraint px*x + py*y = income.

        Prefer *stroke* for line styling; *color*, *linewidth*, and *linestyle*
        are shorthand that a :class:`Stroke` supersedes.

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
        linestyle : str, optional
            Matplotlib line-style string. *None* -> ``theme.budget_stroke``'s style.
        label : str or None
            Optional legend label rendered in LaTeX math mode.
        fill : bool or Fill
            ``True`` shades the feasible set below the budget line; a
            :class:`Fill` also sets its colour and opacity (default
            ``theme.budget_fill``, coloured like the budget line).
        fill_alpha : float or None
            Opacity of the shading. *None* → ``theme.budget_fill_alpha``.
            A ``Fill`` alpha takes precedence.

        stroke : Stroke, optional
            Line style for the budget line (default ``theme.budget_stroke``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        t = self.theme
        shade = (fill if isinstance(fill, Fill) else Fill()).merged_over(
            Fill(alpha=fill_alpha).merged_over(t.budget_fill)
        )
        line_color = color or t.budget_color
        with styled(self, {"budget": stroke}):
            render_budget(
                self.ax,
                px=px,
                py=py,
                income=income,
                color=line_color,
                linewidth=linewidth if linewidth is not None else t.budget_linewidth,
                # Theme.budget_stroke always normalises style to a LineStyle.
                linestyle=linestyle or cast(LineStyle, t.budget_stroke.style).value,
                label=label,
                fill=fill is not False,
                fill_alpha=shade.opacity if shade.opacity is not None else t.budget_fill_alpha,
                # Unset, the fill follows the line, including a Stroke colour.
                fill_color=shade.color or (stroke.color if stroke is not None and stroke.color else None),
            )
        return self

    def add_equilibrium(
        self,
        eq,
        color: str | None = None,
        markersize: float | None = None,
        label: str | Label | None = "x^*",
        drop_dashes: bool = True,
        show_ray: bool = False,
        drop_stroke: Stroke | None = None,
        ray_stroke: Stroke | None = None,
        marker: Marker | None = None,
    ) -> Self:
        """Annotate a pre-solved equilibrium on the canvas.

        Call :func:`~utility_viz.models.optimization.solve` first, then pass the
        resulting :class:`~utility_viz.models.optimization.Equilibrium` here.

        Parameters
        ----------
        eq : Equilibrium
            A solved equilibrium bundle.
        color : str or None
            Marker / drop-line colour. *None* → ``theme.eq_color``.
        markersize : float or None
            Dot size. *None* → ``theme.eq_markersize``.
        label : str, Label, or None
            LaTeX label placed next to the dot, or a :class:`Label` for its
            text, position, colour, and size (default ``theme.point_label``;
            a Label without text keeps ``x^*``). *None* draws no label.
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
        t = self.theme
        text, style = split_label(label, t.point_label, "x^*")
        with styled(
            self,
            {"drop": drop_stroke, "ray": ray_stroke},
            markers={"equilibrium": marker},
            labels={"equilibrium_label": style},
        ):
            # Theme.eq_marker/drop_stroke/ray_stroke always normalise shape/style.
            render_equilibrium(
                self.ax,
                eq=eq,
                color=color or t.eq_color,
                markersize=markersize if markersize is not None else t.eq_markersize,
                label=text,
                drop_dashes=drop_dashes,
                show_ray=show_ray,
                ray_color=t.ray_color,
                ray_linewidth=t.ray_linewidth,
                x_max=self.x_max,
                y_max=self.y_max,
                marker_shape=t.eq_marker.shape or "o",
                drop_linestyle=cast(LineStyle, t.drop_stroke.style).value,
                ray_linestyle=cast(LineStyle, t.ray_stroke.style).value,
            )
        return self

    def add_ray(
        self,
        slope: float,
        color: str | None = None,
        linewidth: float | None = None,
        stroke: Stroke | None = None,
    ) -> Self:
        """Add a dashed ray emanating from the origin.

        Prefer *stroke* for line styling; *color* and *linewidth* are
        shorthand that a :class:`Stroke` supersedes.

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
            from utility_viz.core.diagrams.components import draw_ray

            t = self.theme
            draw_ray(
                self.ax,
                slope,
                self.x_max,
                self.y_max,
                color=color or t.ray_color,
                linewidth=linewidth if linewidth is not None else t.ray_linewidth,
                # Theme.ray_stroke always normalises style to a LineStyle.
                linestyle=cast(LineStyle, t.ray_stroke.style).value,
            )
        return self

    def add_point(
        self,
        x: float,
        y: float,
        label: str | Label | None = None,
        color: str | None = None,
        markersize: float | None = None,
        offset: tuple[float, float] | None = None,
        marker: Marker | None = None,
    ) -> Self:
        """Plot a labelled point on the canvas.

        Parameters
        ----------
        x, y : float
            Coordinates of the point.
        label : str, Label, or None
            Text label (rendered in LaTeX math mode if provided), or a
            :class:`Label` for its text, position, colour, and size (default
            ``theme.point_label``).
        color : str or None
            Marker and label colour. *None* → ``theme.eq_color``.
        markersize : float or None
            Size of the dot. *None* → ``theme.point_marker.size``.
        offset : tuple[float, float], optional
            ``(dx, dy)`` text offset in points from the marker centre
            (default ``(5, 5)``). A Label position or offset takes precedence.

        marker : Marker, optional
            Colour, size, and shape of the point (default ``theme.point_marker``).

        Returns
        -------
        Canvas
            *self*, to allow method chaining.
        """
        text, style = split_label(label, self.theme.point_label)
        moves = isinstance(label, Label) and (label.position is not None or label.offset is not None)
        if offset is not None and not moves:
            # An explicit (dx, dy) keeps its place unless the Label itself moves the text.
            style = replace(style, position=None, offset=None)
        with styled(self, {}, markers={"point": marker}, labels={"point_label": style}):
            c = color or self.theme.eq_color
            plot_point(
                self.ax,
                x=x,
                y=y,
                color=c,
                markersize=(
                    markersize if markersize is not None else (self.theme.point_marker.size or self.theme.eq_markersize)
                ),
                marker=self.theme.point_marker.shape or "o",
                linestyle="None",
                zorder=6,
                clip_on=False,
                role="point",
            )
            if text:
                annotate_math(
                    self.ax,
                    x=x,
                    y=y,
                    text=text,
                    color=c,
                    offset=offset or (5, 5),
                    fontsize=12,
                    zorder=7,
                    role="point_label",
                    default=None if offset else Label(position=LabelPosition.TOP_RIGHT, offset=5),
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
    ) -> Self:
        """Draw a PCC/ICC-style path through a sequence of equilibria.

        Prefer *stroke* for line styling; *color* and *linewidth* are
        shorthand that a :class:`Stroke` supersedes.

        Parameters
        ----------
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
        with styled(
            self,
            {"path": stroke, "budget": budget_stroke, "curve": curve_stroke},
            markers={"path_point": point_marker, "equilibrium": equilibrium_marker},
        ):
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

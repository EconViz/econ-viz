"""Path renderer for Canvas PCC/ICC-like overlays."""

from __future__ import annotations

import matplotlib.lines as mlines


def render_path(
    *,
    canvas,
    path,
    color: str,
    linewidth: float,
    label: str | None,
    show_points: bool,
    show_budgets: bool,
    show_curves: bool,
    show_equilibria: bool,
    invert_axes: bool,
    smooth_curve: bool,
    smooth_fn,
    extend_fn,
) -> None:
    """Render a solved consumption path on a canvas."""
    xs = list(path.x_values if not invert_axes else path.parameter_values)
    ys = list(path.y_values if not invert_axes else path.x_values)

    curve_xs, curve_ys = (xs, ys)
    if smooth_curve:
        curve_xs, curve_ys = smooth_fn(xs, ys)
        curve_xs, curve_ys = extend_fn(curve_xs, curve_ys)
    canvas.ax.plot(curve_xs, curve_ys, color=color, linewidth=linewidth, clip_on=False)

    if label:
        canvas._legend_handles.append(mlines.Line2D([], [], color=color, linewidth=linewidth, label=label))

    for idx, eq in enumerate(path.equilibria):
        if show_curves and hasattr(path, "func"):
            canvas.add_utility(path.func, levels=[eq.utility], color=color, linewidth=max(linewidth * 0.75, 0.8))
        if show_budgets:
            canvas.add_budget(
                path.px_values[idx],
                path.py_values[idx],
                path.income_values[idx],
                color=color,
                linewidth=max(linewidth * 0.6, 0.8),
                linestyle=":",
            )
        if show_equilibria:
            canvas.add_equilibrium(eq, color=color, label=None, drop_dashes=False)
        elif show_points:
            px, py = (eq.x, eq.y) if not invert_axes else (path.parameter_values[idx], eq.x)
            canvas.ax.plot(px, py, "o", color=color, markersize=max(canvas.theme.eq_markersize - 1, 3))


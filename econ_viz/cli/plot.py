"""``plot`` sub-command — generate and export an economic diagram."""

from __future__ import annotations

import argparse
import sys

from .resolve import build_model, resolve_theme


def cmd_plot(args: argparse.Namespace) -> None:
    """Run the ``plot`` sub-command.

    Builds a :class:`~econ_viz.Canvas`, optionally adds indifference
    curves, a budget line, and an equilibrium point, then either saves
    the figure to *args.output* or opens an interactive window.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments produced by :func:`~econ_viz.cli.main.build_parser`.
    """
    from econ_viz import Canvas, levels as lvl_mod, solve

    if args.model is None and args.latex is None:
        print("error: provide --model <name> or --latex <expr>", file=sys.stderr)
        sys.exit(1)

    model = build_model(args)
    theme = resolve_theme(args.theme)

    cvs = Canvas(
        x_max=args.x_max,
        y_max=args.y_max,
        x_label=args.x_label,
        y_label=args.y_label,
        title=args.title,
        dpi=args.dpi,
        theme=theme,
    )

    eq = _maybe_solve(args, model, solve)

    if not args.no_curves:
        lvls = _compute_levels(args, model, eq, lvl_mod)
        cvs.add_utility(model, levels=lvls)

    if not args.no_budget and _has_prices(args):
        cvs.add_budget(args.px, args.py, args.income, fill=args.fill)

    if not args.no_equilibrium and eq is not None:
        cvs.add_equilibrium(eq, show_ray=args.show_ray)

    if args.output:
        cvs.save(args.output)
        print(f"Saved to {args.output}")
    else:
        cvs.show()


def _has_prices(args: argparse.Namespace) -> bool:
    """Return True when px, py, and income are all provided."""
    return args.px is not None and args.py is not None and args.income is not None


def _maybe_solve(args: argparse.Namespace, model, solve_fn):
    """Solve for the consumer optimum when prices and income are available.

    Returns ``None`` when prices are missing or ``--no-equilibrium`` is set.
    """
    if args.no_equilibrium or not _has_prices(args):
        return None
    return solve_fn(model, px=args.px, py=args.py, income=args.income)


def _compute_levels(args: argparse.Namespace, model, eq, lvl_mod) -> list[float]:
    """Compute indifference-curve utility levels.

    Uses :func:`~econ_viz.levels.around` when an equilibrium is available,
    otherwise falls back to :func:`~econ_viz.levels.percentile`.
    """
    if eq is not None:
        return lvl_mod.around(eq.utility, n=args.n_curves)

    import numpy as np

    x_pts = np.linspace(1e-3, args.x_max, 200)
    y_pts = np.linspace(1e-3, args.y_max, 200)
    X, Y = np.meshgrid(x_pts, y_pts)
    Z = model(X, Y)
    return lvl_mod.percentile(Z, n=args.n_curves)

"""CLI entry point and argument parser for econ-viz."""

from __future__ import annotations

import argparse

from .help import cmd_help
from .models import cmd_models
from .plot import cmd_plot


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """Construct the top-level parser and all sub-command parsers.

    Returns
    -------
    parser : argparse.ArgumentParser
        The root parser.
    subparsers : dict[str, argparse.ArgumentParser]
        Mapping of sub-command name to its individual parser, for use by
        the ``help`` command.
    """
    parser = argparse.ArgumentParser(
        prog="econ-viz",
        description="Produce publication-quality microeconomics diagrams.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    subparsers: dict[str, argparse.ArgumentParser] = {}

    subparsers["models"] = sub.add_parser("models", help="List available utility models.")
    subparsers["plot"]   = _register_plot(sub)
    subparsers["help"]   = _register_help(sub)

    return parser, subparsers


def _register_help(sub) -> argparse.ArgumentParser:
    """Add the ``help`` sub-command to *sub* and return its parser."""
    p = sub.add_parser("help", help="Show help for a command.")
    p.add_argument(
        "topic",
        nargs="?",
        metavar="<command>",
        help="Command to get help on: models, plot.",
    )
    return p


def _register_plot(sub) -> argparse.ArgumentParser:
    """Add the ``plot`` sub-command to *sub* and return its parser."""
    p = sub.add_parser("plot", help="Generate a diagram and save or display it.")

    mgrp = p.add_mutually_exclusive_group()
    mgrp.add_argument(
        "--model", "-m",
        metavar="NAME",
        help="Model name: cobb-douglas, leontief, perfect-substitutes, ces, satiation.",
    )
    mgrp.add_argument(
        "--latex", "-l",
        metavar="EXPR",
        help=r'LaTeX expression, e.g. "x^{0.5} y^{0.5}" or "\min(2x, 3y)".',
    )

    p.add_argument("--alpha",   type=float, metavar="A", help="Alpha parameter.")
    p.add_argument("--beta",    type=float, metavar="B", help="Beta parameter.")
    p.add_argument("--a",       type=float, metavar="A", help="a parameter (Leontief / PerfectSubs / Satiation).")
    p.add_argument("--b",       type=float, metavar="B", help="b parameter (Leontief / PerfectSubs / Satiation).")
    p.add_argument("--rho",     type=float, metavar="R", help="Rho parameter (CES).")
    p.add_argument("--bliss-x", dest="bliss_x", type=float, metavar="X", help="Bliss point x (Satiation).")
    p.add_argument("--bliss-y", dest="bliss_y", type=float, metavar="Y", help="Bliss point y (Satiation).")

    p.add_argument("--px",     type=float, metavar="P", help="Price of good x.")
    p.add_argument("--py",     type=float, metavar="P", help="Price of good y.")
    p.add_argument("--income", type=float, metavar="I", help="Consumer income.")

    p.add_argument("--x-max",   dest="x_max",   type=float, default=10.0, help="Horizontal axis limit (default 10).")
    p.add_argument("--y-max",   dest="y_max",   type=float, default=10.0, help="Vertical axis limit (default 10).")
    p.add_argument("--x-label", dest="x_label", type=str,   default="x",  help="Horizontal axis label (default 'x').")
    p.add_argument("--y-label", dest="y_label", type=str,   default="y",  help="Vertical axis label (default 'y').")
    p.add_argument("--title",   type=str,   default=None, help="Figure title.")
    p.add_argument("--dpi",     type=int,   default=300,  help="Output DPI for raster images (default 300).")
    p.add_argument("--theme",   type=str,   default="default", metavar="NAME",
                   help="Theme name: default, nord (default: default).")

    p.add_argument("--n-curves",       dest="n_curves",       type=int, default=5,
                   help="Number of indifference curves (default 5).")
    p.add_argument("--fill",           action="store_true", help="Shade the feasible set under the budget line.")
    p.add_argument("--show-ray",       dest="show_ray",      action="store_true",
                   help="Draw expansion-path ray through the optimum.")
    p.add_argument("--no-budget",      dest="no_budget",      action="store_true", help="Omit the budget line.")
    p.add_argument("--no-equilibrium", dest="no_equilibrium", action="store_true", help="Omit the equilibrium point.")
    p.add_argument("--no-curves",      dest="no_curves",      action="store_true", help="Omit indifference curves.")

    p.add_argument("--output", "-o", metavar="FILE",
                   help="Output file (.png, .pdf, .svg, .tex). Omit to display interactively.")

    return p


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command."""
    parser, subparsers = build_parser()
    args = parser.parse_args()

    if args.command == "help":
        cmd_help(args, parser, subparsers)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "plot":
        cmd_plot(args)

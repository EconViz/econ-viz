"""Model and theme resolution helpers for the CLI."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_model(args: argparse.Namespace):
    """Instantiate the utility model specified by CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Either ``args.latex`` or ``args.model``
        must be set.

    Returns
    -------
    UtilityFunction
        A concrete model conforming to the ``UtilityFunction`` protocol.
    """
    from econ_viz.models import (
        CES,
        CobbDouglas,
        Leontief,
        PerfectSubstitutes,
        QuasiLinear,
        Satiation,
        StoneGeary,
        Translog,
    )
    from econ_viz.parser import parse_latex

    if args.latex:
        return parse_latex(args.latex)

    name = (args.model or "").lower()

    if name == "cobb-douglas":
        return CobbDouglas(
            alpha=args.alpha if args.alpha is not None else 0.5,
            beta=args.beta   if args.beta  is not None else 0.5,
        )
    if name == "leontief":
        return Leontief(
            a=args.a if args.a is not None else 1.0,
            b=args.b if args.b is not None else 1.0,
        )
    if name == "perfect-substitutes":
        return PerfectSubstitutes(
            a=args.a if args.a is not None else 1.0,
            b=args.b if args.b is not None else 1.0,
        )
    if name == "ces":
        return CES(
            rho=args.rho     if args.rho   is not None else 0.5,
            alpha=args.alpha if args.alpha is not None else 0.5,
            beta=args.beta   if args.beta  is not None else 0.5,
        )
    if name == "satiation":
        return Satiation(
            bliss_x=args.bliss_x if args.bliss_x is not None else 5.0,
            bliss_y=args.bliss_y if args.bliss_y is not None else 5.0,
            a=args.a if args.a is not None else 1.0,
            b=args.b if args.b is not None else 1.0,
        )
    if name == "quasi-linear":
        return QuasiLinear(
            v_func=_resolve_v_func(args.v_func),
            linear_in=args.linear_in if args.linear_in is not None else "y",
        )
    if name == "stone-geary":
        return StoneGeary(
            alpha=args.alpha if args.alpha is not None else 0.5,
            beta=args.beta if args.beta is not None else 0.5,
            bar_x=args.bar_x if args.bar_x is not None else 1.0,
            bar_y=args.bar_y if args.bar_y is not None else 1.0,
        )
    if name == "translog":
        return Translog(
            alpha_0=args.alpha_0 if args.alpha_0 is not None else 0.0,
            alpha_x=args.alpha_x if args.alpha_x is not None else 0.5,
            alpha_y=args.alpha_y if args.alpha_y is not None else 0.5,
            beta_xx=args.beta_xx if args.beta_xx is not None else 0.0,
            beta_yy=args.beta_yy if args.beta_yy is not None else 0.0,
            beta_xy=args.beta_xy if args.beta_xy is not None else 0.0,
        )

    print(
        f"error: unknown model '{args.model}'. "
        "Run `econ-viz models` to see available models.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_v_func(name: str | None):
    """Map a CLI v_func name to a NumPy callable."""
    if name in (None, "log"):
        return np.log
    if name == "sqrt":
        return np.sqrt
    print(
        f"error: unknown QuasiLinear v_func '{name}'. Available: log, sqrt",
        file=sys.stderr,
    )
    sys.exit(1)


_THEMES: dict[str, object] | None = None


def _themes_map() -> dict[str, object]:
    """Return the name→Theme mapping, loading themes lazily."""
    global _THEMES
    if _THEMES is None:
        from econ_viz import themes
        _THEMES = {
            "default": themes.default,
            "nord":    themes.nord,
        }
    return _THEMES


def resolve_theme(name: str):
    """Look up a theme by name.

    Parameters
    ----------
    name : str
        Theme name (case-insensitive).

    Returns
    -------
    Theme
        The matching theme instance.
    """
    mapping = _themes_map()
    theme = mapping.get(name.lower())
    if theme is None:
        available = ", ".join(mapping)
        print(f"error: unknown theme '{name}'. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return theme

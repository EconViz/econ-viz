"""Model and theme resolution helpers for the CLI."""

from __future__ import annotations

import argparse
import sys


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
    from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES, Satiation
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

    print(
        f"error: unknown model '{args.model}'. "
        "Run `econ-viz models` to see available models.",
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

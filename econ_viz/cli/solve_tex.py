"""``solve-tex`` sub-command — print closed-form Marshallian demand in TeX."""

from __future__ import annotations

import argparse
import sys

from .resolve import build_model


def cmd_solve_tex(args: argparse.Namespace) -> None:
    """Print a closed-form Marshallian demand formula as plain TeX text."""
    from econ_viz import solution_tex

    if args.model is None and args.latex is None:
        print("error: provide --model <name> or --latex <expr>", file=sys.stderr)
        sys.exit(1)

    model = build_model(args)
    try:
        tex = solution_tex(
            model,
            px=args.px_symbol,
            py=args.py_symbol,
            income=args.income_symbol,
            symbolic_params=args.symbolic_params,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(tex)

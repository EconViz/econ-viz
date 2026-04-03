"""``solve-tex`` sub-command — print closed-form Marshallian demand in TeX."""

from __future__ import annotations

import argparse

from .errors import CliConfigError
from .resolve import build_model


def cmd_solve_tex(args: argparse.Namespace) -> None:
    """Print a closed-form Marshallian demand formula as plain TeX text."""
    from econ_viz import solution_tex

    if args.model is None and args.latex is None:
        raise CliConfigError("provide --model <name> or --latex <expr>")

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
        raise CliConfigError(str(exc)) from exc

    print(tex)

"""``models`` sub-command — list available utility models."""

from __future__ import annotations

import argparse


_MODEL_ENTRIES = [
    ("cobb-douglas",        "CobbDouglas         --alpha A --beta B"),
    ("leontief",            "Leontief            --a A --b B"),
    ("perfect-substitutes", "PerfectSubstitutes  --a A --b B"),
    ("ces",                 "CES                 --rho R [--alpha A --beta B]"),
    ("satiation",           "Satiation           --bliss-x X --bliss-y Y [--a A --b B]"),
    ("quasi-linear",        "QuasiLinear         [--v-func log|sqrt] [--linear-in x|y]"),
    ("stone-geary",         "StoneGeary          [--alpha A --beta B --bar-x X --bar-y Y]"),
    ("translog",            "Translog            [--alpha-0 A0 --alpha-x AX --alpha-y AY --beta-xx BXX --beta-yy BYY --beta-xy BXY]"),
]

_LATEX_EXAMPLES = [
    r'--latex "x^{0.4} y^{0.6}"',
    r'--latex "\min(2x, 3y)"',
    r'--latex "2x + 3y"',
]


def cmd_models(_args: argparse.Namespace) -> None:
    """Print a formatted list of all supported models and their parameters."""
    print("Available models:\n")
    for _, description in _MODEL_ENTRIES:
        print(f"  {description}")
    print()
    print("LaTeX shortcut (Cobb-Douglas / Leontief / Perfect Substitutes):")
    for example in _LATEX_EXAMPLES:
        print(f"  {example}")

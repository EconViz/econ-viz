"""``models`` sub-command — list available utility models."""

from __future__ import annotations

import argparse

from econ_viz.models import get_model_registry

_MODEL_FLAGS = {
    "cobb-douglas": "--alpha A --beta B",
    "leontief": "--a A --b B",
    "perfect-substitutes": "--a A --b B",
    "ces": "--rho R [--alpha A --beta B]",
    "satiation": "--bliss-x X --bliss-y Y [--a A --b B]",
    "quasi-linear": "[--v-func log|sqrt] [--linear-in x|y]",
    "stone-geary": "[--alpha A --beta B --bar-x X --bar-y Y]",
    "translog": "[--alpha-0 A0 --alpha-x AX --alpha-y AY --beta-xx BXX --beta-yy BYY --beta-xy BXY]",
}

_MODEL_LABELS = {
    "cobb-douglas": "CobbDouglas",
    "leontief": "Leontief",
    "perfect-substitutes": "PerfectSubstitutes",
    "ces": "CES",
    "satiation": "Satiation",
    "quasi-linear": "QuasiLinear",
    "stone-geary": "StoneGeary",
    "translog": "Translog",
}

_LATEX_EXAMPLES = [
    r'--latex "x^{0.4} y^{0.6}"',
    r'--latex "\min(2x, 3y)"',
    r'--latex "2x + 3y"',
]


def cmd_models(_args: argparse.Namespace) -> None:
    """Print a formatted list of all supported models and their parameters."""
    print("Available models:\n")
    for model_name in sorted(get_model_registry().keys()):
        label = _MODEL_LABELS.get(model_name, model_name)
        flags = _MODEL_FLAGS.get(model_name, "")
        print(f"  {label:<19} {flags}".rstrip())
    print()
    print("LaTeX shortcut (Cobb-Douglas / Leontief / Perfect Substitutes):")
    for example in _LATEX_EXAMPLES:
        print(f"  {example}")

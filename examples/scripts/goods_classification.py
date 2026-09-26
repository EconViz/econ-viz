"""Hicks decompositions of a fall in p_x for five kinds of goods.

Normal and income-neutral goods use Cobb-Douglas and quasi-linear utility.
The three inferior cases use Haagsma's (2012) utility

    U(x, y) = a ln(x - g) - b ln(d - y),   b > a,

whose demand x = a (d p_y - I) / ((b - a) p_x) + b g / (b - a) is inferior in
income. Setting d p_y above, equal to, or below income I gives an ordinary
inferior good, exactly offsetting effects, and a Giffen good.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from econ_viz import Canvas, Effect
from econ_viz.models import CobbDouglas, QuasiLinear
from econ_viz.models.advanced import CustomUtility
from econ_viz.optimizer import DecompositionMethod, decompose_price_effect

OUTPUT_DIR = Path("examples/output/decom/goods")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PX = (2.0, 1.0)  # p_x falls
PY = 1.0
INCOME = 28.0


def haagsma(d: float, a: float = 1.0, b: float = 2.0, g: float = 2.0) -> CustomUtility:
    """Haagsma utility, defined for x > g and y < d."""

    def utility(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((x > g) & (y < d), a * np.log(x - g) - b * np.log(d - y), np.nan)

    return CustomUtility(func=utility, name=f"Haagsma(d={d:g})")


CASES = [
    ("1_normal", "Normal good", CobbDouglas(alpha=0.5, beta=0.5)),
    ("2_neutral", "Income-neutral good", QuasiLinear(v_func=lambda z: 4.0 * np.log(z), linear_in="y")),
    ("3_inferior", "Inferior good: SE > |IE|", haagsma(d=30.0)),
    ("4_offsetting", "Inferior good: SE = |IE|", haagsma(d=28.0)),
    ("5_giffen", "Giffen good: SE < |IE|", haagsma(d=27.0)),
]


def _axis_limits(dec) -> tuple[float, float]:
    xs = [dec.A.x, dec.B.x, dec.C.x, dec.income / dec.px_after, dec.compensated_income / dec.px_after]
    ys = [dec.A.y, dec.B.y, dec.C.y, dec.income / dec.py, dec.compensated_income / dec.py]
    return max(xs) * 1.2, max(ys) * 1.25


def build_examples() -> None:
    for slug, title, model in CASES:
        with warnings.catch_warnings():
            # Inferior and Giffen goods are the point here; silence the library's reminders.
            warnings.filterwarnings("ignore", message=".*(inferior|Giffen) good.*")
            dec = decompose_price_effect(model, px=PX, py=PY, income=INCOME, method=DecompositionMethod.HICKS)
        x_max, y_max = _axis_limits(dec)
        (
            Canvas(x_max=x_max, y_max=y_max, x_label="x", y_label="y", title=title)
            .add_decomposition(
                dec,
                show_x_projections=True,
                substitution=Effect(label="SE"),
                income=Effect(label="IE"),
            )
            .save(str(OUTPUT_DIR / f"{slug}.png"))
        )
        se, ie, te = dec.substitution_effect[0], dec.income_effect[0], dec.total_effect[0]
        print(f"[ok] {slug}: SE={se:+.2f} IE={ie:+.2f} TE={te:+.2f}")


if __name__ == "__main__":
    build_examples()

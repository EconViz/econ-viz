<p align="center">
  <img src="https://raw.githubusercontent.com/EconViz/econ-viz-docs/main/docs/assets/banner.svg" alt="Econ-Viz" width="480">
</p>

<p align="center">
  <a href="https://pypi.org/project/econ-viz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <a href="https://pypi.org/project/econ-viz/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-181818?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-235%20passed-181818?style=flat-square&color=181818&labelColor=f3f3f3">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-99%25-181818?style=flat-square&color=181818&labelColor=f3f3f3">
</p>

A Python toolkit for producing publication-quality microeconomics diagrams. Define utility functions declaratively, solve for consumer equilibria, and export figures as PNG, PDF, or SVG — all in a few lines of code.

## Installation

```bash
pip install econ-viz
```

Requires Python 3.12 or later.

## Quick Start

```python
from econ_viz import Canvas, levels, solve
from econ_viz.models import CobbDouglas

model = CobbDouglas(alpha=0.5, beta=0.5)
eq    = solve(model, px=2.0, py=3.0, income=30.0)
lvls  = levels.around(eq.utility, n=5)

cvs = Canvas(x_max=20, y_max=15, x_label="x", y_label="y",
             title="Cobb-Douglas  $x^{0.5} y^{0.5}$")
cvs.add_utility(model, levels=lvls)
cvs.add_budget(2.0, 3.0, 30.0, fill=True)
cvs.add_equilibrium(eq, show_ray=True)
cvs.save("cobb_douglas.png")
```

![Cobb-Douglas indifference map with budget line and equilibrium point](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/cobb_douglas_eq.png)

## Notebook

The project ships with an interactive playground notebook:

[`notebook/econ-viz Playground.ipynb`](notebook/econ-viz%20Playground.ipynb)

Download it and open it in Jupyter, VS Code, or Colab. The first code cell upgrades `econ-viz` from PyPI for fresh runtimes.

## Highlights

- Built-in models: Cobb-Douglas, Leontief, Perfect Substitutes, CES, Satiation, Quasi-Linear, Stone-Geary, and Translog
- Solver support for interior, kink, boundary, and corner solutions
- Closed-form demand helpers with `solution_tex(...)`
- Comparative tools including `comparative_statics(...)` and `slutsky_matrix(...)`
- Multi-panel `Figure` layouts, `PricePath` / `IncomePath`, and linked `DemandDiagram`
- CLI support for plotting and closed-form demand output

## Additional Tools

Closed-form Marshallian demand in TeX:

```python
from econ_viz import solution_tex
from econ_viz.models import CobbDouglas

tex = solution_tex(CobbDouglas(alpha=0.4, beta=0.6))
```

Slutsky matrix:

```python
from econ_viz import slutsky_matrix
from econ_viz.models import CobbDouglas

S = slutsky_matrix(CobbDouglas(alpha=0.4, beta=0.6), px=2.0, py=3.0, income=60.0)
# S.s_xx, S.s_xy, S.s_yx, S.s_yy
```

## CLI

```bash
econ-viz help
econ-viz models
econ-viz solve-tex --model cobb-douglas --symbolic-params
```

Plotting example:

```bash
econ-viz plot --model cobb-douglas --alpha 0.5 --beta 0.5 \
              --px 2 --py 3 --income 30 \
              --fill --show-ray \
              --output cobb_douglas.png
```

## Documentation

Full documentation lives at [econ-viz.org](https://econ-viz.org).

## License

MIT © Anthony Sung

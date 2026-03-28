
<p align="center">
  <img src="https://raw.githubusercontent.com/EconViz/econ-viz-docs/main/docs/assets/banner.svg" alt="Econ-Viz" width="480">
</p>

<p align="center">
  <a href="https://pypi.org/project/econ-viz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <a href="https://pypi.org/project/econ-viz/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-181818?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-183%20passed-181818?style=flat-square&color=181818&labelColor=f3f3f3">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-99%25-181818?style=flat-square&color=181818&labelColor=f3f3f3">
</p>

A Python toolkit for producing publication-quality microeconomics diagrams. Define utility functions declaratively, solve for consumer equilibria, and export figures as raster images or LaTeX/TikZ source — all in a few lines of code.

<p align="center">
  <a href="https://colab.research.google.com/drive/10SVRHL3UTASF5nKroxzYcsVRC5of7YQD">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>

> Open the notebook in Colab, then go to **File → Save a copy in Drive** to keep your own editable version.

## Installation

```bash
pip install econ-viz
```

Requires Python 3.12 or later.

## Quick Start

```python
import numpy as np
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

## CLI

`econ-viz` ships with a command-line interface for generating diagrams without writing Python.

### Commands

| Command | Description |
|---------|-------------|
| `econ-viz help [<command>]` | Show help for the CLI or a specific command |
| `econ-viz models` | List all supported utility models |
| `econ-viz plot ...` | Generate and export a diagram |

### Examples

```bash
# Cobb-Douglas with equilibrium and budget line
econ-viz plot --model cobb-douglas --alpha 0.5 --beta 0.5 \
              --px 2 --py 3 --income 30 \
              --output cobb_douglas.png

# Parse a LaTeX expression directly
econ-viz plot --latex "x^{0.4} y^{0.6}" \
              --px 2 --py 3 --income 30 \
              --output cd_latex.png

# Leontief with Nord theme and expansion-path ray
econ-viz plot --model leontief --a 1 --b 2 \
              --px 2 --py 3 --income 30 \
              --theme nord --show-ray \
              --output leontief.png

# CES — indifference curves only, no budget or equilibrium
econ-viz plot --model ces --rho -0.5 \
              --x-max 20 --y-max 15 --n-curves 6 \
              --no-budget --no-equilibrium \
              --output ces.png

# Omit --output to open an interactive window
econ-viz plot --model cobb-douglas --px 2 --py 3 --income 30
```

### `plot` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | — | Model name (see `econ-viz models`) |
| `--latex`, `-l` | — | LaTeX expression (Cobb-Douglas / Leontief / Perfect Substitutes) |
| `--px`, `--py`, `--income` | — | Prices and budget |
| `--alpha`, `--beta` | 0.5 | Cobb-Douglas / CES share parameters |
| `--a`, `--b` | 1.0 | Leontief / Perfect Substitutes / Satiation coefficients |
| `--rho` | 0.5 | CES substitution parameter |
| `--bliss-x`, `--bliss-y` | 5.0 | Satiation bliss point |
| `--x-max`, `--y-max` | 10 | Canvas axis limits |
| `--x-label`, `--y-label` | `x`, `y` | Axis labels |
| `--title` | — | Figure title |
| `--theme` | `default` | Colour theme: `default`, `nord` |
| `--n-curves` | 5 | Number of indifference curves |
| `--dpi` | 300 | Raster output resolution |
| `--fill` | off | Shade feasible set below the budget line |
| `--show-ray` | off | Draw expansion-path ray through the optimum |
| `--no-budget` | off | Omit the budget line |
| `--no-equilibrium` | off | Omit the equilibrium point |
| `--no-curves` | off | Omit indifference curves |
| `--output`, `-o` | — | Output file; omit to open an interactive window |

## Utility Models

### Cobb-Douglas

```python
from econ_viz.models import CobbDouglas

model = CobbDouglas(alpha=0.3, beta=0.7)
```

![Cobb-Douglas indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/cobb_douglas.png)

### Leontief (Perfect Complements)

```python
from econ_viz.models import Leontief

model = Leontief(a=1.0, b=1.0)   # U = min(ax, by)
```

![Leontief indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/leontief_eq.png)

### Perfect Substitutes

```python
from econ_viz.models import PerfectSubstitutes

model = PerfectSubstitutes(a=1.0, b=2.0)   # U = ax + by
```

![Perfect substitutes indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/perfect_substitutes_eq.png)

### CES

```python
from econ_viz.models import CES

model = CES(rho=-0.5, alpha=0.5)   # elasticity of substitution = 1/(1+rho)
```

![CES indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/ces_eq.png)

### Satiation (Bliss Point)

```python
from econ_viz.models import Satiation

model = Satiation(bliss_x=6.0, bliss_y=4.0, a=1.0, b=1.0)
```

![Satiation indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/satiation.png)

### Quasi-Linear

```python
import numpy as np
from econ_viz.models import QuasiLinear

model = QuasiLinear(v_func=np.log, linear_in="y")   # U = log(x) + y
```

![Quasi-linear indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/quasi_linear.png)

## LaTeX Input

Parse standard LaTeX math expressions directly into model instances:

```python
from econ_viz import parse_latex

cd  = parse_latex(r"x^{0.4} y^{0.6}")
leo = parse_latex(r"\min(2x, 3y)")
ps  = parse_latex(r"2x + 3y")
```

The parser accepts common preambles such as `U(x,y) =`, `U =`, and bare expressions. Unrecognised patterns raise `ParseError`.

![Parsed Cobb-Douglas from LaTeX](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/latex_cobb_douglas_u.png)

## Advanced Models

### Custom Utility

Wrap any vectorised Python callable as a first-class model. The callable is validated at construction time against a random NumPy mesh-grid.

```python
import numpy as np
from econ_viz.models import CustomUtility

model = CustomUtility(func=lambda x, y: np.log(x) + np.log(y), name="log+log")
```

![Custom utility indifference map](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/advanced_custom.png)

### Multi-Good Cobb-Douglas

Model preferences over N goods and project to a 2-D canvas via `freeze()`:

```python
from econ_viz.models import MultiGoodCD

m3   = MultiGoodCD({'x': 0.3, 'y': 0.3, 'z': 0.4})
flat = m3.freeze(z=10.0)   # returns a CustomUtility ready for Canvas
```

```python
from econ_viz import Canvas, levels, solve

eq   = solve(flat, px=2.0, py=3.0, income=30.0)
lvls = levels.around(eq.utility, n=5)

cvs = Canvas(x_max=20, y_max=15, title=r"MultiGoodCD  $z=10$")
cvs.add_utility(flat, levels=lvls)
cvs.add_budget(2.0, 3.0, 30.0, fill=True)
cvs.add_equilibrium(eq)
cvs.save("multigood.png")
```

![Multi-good Cobb-Douglas frozen slice](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/advanced_multigd.png)

## Solving for Equilibrium

`solve()` returns an `Equilibrium` named tuple with fields `x`, `y`, and `utility`:

```python
from econ_viz import solve
from econ_viz.models import CobbDouglas

eq = solve(CobbDouglas(), px=2.0, py=3.0, income=30.0)
print(eq.x, eq.y, eq.utility)
```

## Themes

```python
from econ_viz import Canvas, themes

cvs = Canvas(x_max=20, y_max=15)                    # default theme
cvs = Canvas(x_max=20, y_max=15, theme=themes.nord) # nord theme
```

| Default | Nord |
|---------|------|
| ![Default theme](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/theme_default.png) | ![Nord theme](https://raw.githubusercontent.com/EconViz/econ-viz/a8423043789ee7dba19b2d71fa6cc5071601181a/theme_nord.png) |

## Export

```python
cvs.save("figure.png")    # raster (DPI controlled by Canvas(dpi=300))
cvs.save("figure.tex")    # TikZ/PGFPlots source for LaTeX
```

## License

MIT © Anthony Sung

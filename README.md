
# econ-viz

[![PyPI version](https://img.shields.io/pypi/v/econ-viz.svg)](https://pypi.org/project/econ-viz/)
[![Python](https://img.shields.io/pypi/pyversions/econ-viz.svg)](https://pypi.org/project/econ-viz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-183%20passed-brightgreen)](https://github.com/EconViz/econ-viz/actions)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/EconViz/econ-viz/actions)

A Python toolkit for producing publication-quality microeconomics diagrams. Define utility functions declaratively, solve for consumer equilibria, and export figures as raster images or LaTeX/TikZ source — all in a few lines of code.

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

![Cobb-Douglas indifference map with budget line and equilibrium point](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/cobb_douglas_eq.png)

## Utility Models

### Cobb-Douglas

```python
from econ_viz.models import CobbDouglas

model = CobbDouglas(alpha=0.3, beta=0.7)
```

![Cobb-Douglas indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/cobb_douglas.png)

### Leontief (Perfect Complements)

```python
from econ_viz.models import Leontief

model = Leontief(a=1.0, b=1.0)   # U = min(ax, by)
```

![Leontief indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/leontief_eq.png)

### Perfect Substitutes

```python
from econ_viz.models import PerfectSubstitutes

model = PerfectSubstitutes(a=1.0, b=2.0)   # U = ax + by
```

![Perfect substitutes indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/perfect_substitutes_eq.png)

### CES

```python
from econ_viz.models import CES

model = CES(rho=-0.5, alpha=0.5)   # elasticity of substitution = 1/(1+rho)
```

![CES indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/ces_eq.png)

### Satiation (Bliss Point)

```python
from econ_viz.models import Satiation

model = Satiation(bliss_x=6.0, bliss_y=4.0, a=1.0, b=1.0)
```

![Satiation indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/satiation.png)

### Quasi-Linear

```python
import numpy as np
from econ_viz.models import QuasiLinear

model = QuasiLinear(v_func=np.log, linear_in="y")   # U = log(x) + y
```

![Quasi-linear indifference curves](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/quasi_linear.png)

## LaTeX Input

Parse standard LaTeX math expressions directly into model instances:

```python
from econ_viz import parse_latex

cd  = parse_latex(r"x^{0.4} y^{0.6}")
leo = parse_latex(r"\min(2x, 3y)")
ps  = parse_latex(r"2x + 3y")
```

The parser accepts common preambles such as `U(x,y) =`, `U =`, and bare expressions. Unrecognised patterns raise `ParseError`.

![Parsed Cobb-Douglas from LaTeX](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/latex_cobb_douglas_u.png)

## Advanced Models

### Custom Utility

Wrap any vectorised Python callable as a first-class model. The callable is validated at construction time against a random NumPy mesh-grid.

```python
import numpy as np
from econ_viz.models import CustomUtility

model = CustomUtility(func=lambda x, y: np.log(x) + np.log(y), name="log+log")
```

![Custom utility indifference map](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/advanced_custom.png)

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

![Multi-good Cobb-Douglas frozen slice](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/advanced_multigd.png)

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
| ![Default theme](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/theme_default.png) | ![Nord theme](https://raw.githubusercontent.com/EconViz/econ-viz/45540774ef9c76db769701cf95fa7d3f8d5ae1d2/theme_nord.png) |

## Export

```python
cvs.save("figure.png")    # raster (DPI controlled by Canvas(dpi=300))
cvs.save("figure.tikz")   # TikZ/PGFPlots source for LaTeX
```

## License

MIT © Anthony Sung

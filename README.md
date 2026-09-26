<p align="center">
  <img src="https://raw.githubusercontent.com/EconViz/econ-viz-docs/main/docs/assets/banner.svg" alt="Econ-Viz" width="480">
</p>

<p align="center">
  <a href="https://pypi.org/project/econ-viz/"><img alt="PyPI" src="https://img.shields.io/pypi/v/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3&cacheSeconds=300"></a>
  <a href="https://pypi.org/project/econ-viz/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/econ-viz?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-181818?style=flat-square&color=181818&labelColor=f3f3f3"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-557%20passed-181818?style=flat-square&color=181818&labelColor=f3f3f3">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-92.63%25-181818?style=flat-square&color=181818&labelColor=f3f3f3">
</p>

A Python toolkit for producing publication-quality microeconomics diagrams. Define utility functions declaratively, solve for consumer equilibria, and export figures as PNG, PDF, SVG, or pure TikZ — all in a few lines of code.

## Installation

```bash
pip install econ-viz
```

Requires Python 3.10 or later.

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

TikZ export writes a standalone LaTeX document with only TikZ drawing commands:

```python
cvs.save("cobb_douglas.tex", tikz_scale=0.0125)
```

The default TikZ scale maps a 6 inch wide Matplotlib figure to about 7.5 cm.

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
- Color-blind-friendly default palette (`themes.COLORBLIND_CYCLE_RGB`) sourced from [thriveth/8560036](https://gist.github.com/thriveth/8560036), with related citation at [DOI:10.1080/00220485.1996.10844911](https://www.tandfonline.com/doi/abs/10.1080/00220485.1996.10844911)

## Additional Tools

Axis labels can be placed around their arrowheads, and each axis can use its
own arrowhead style and line style (solid, dashed, dotted, or dashdot):

```python
from econ_viz import ArrowStyle, Canvas, LabelPosition, LineStyle

canvas = Canvas(
    x_label_pos=LabelPosition.TOP,
    y_label_pos=LabelPosition.RIGHT,
    x_arrow_style=ArrowStyle.SIMPLE,
    y_arrow_style=ArrowStyle.WEDGE,
    x_line_style=LineStyle.DASHED,
)
```

Every line can be restyled with a `Stroke`: width, line style, colour, and an
arrowhead at its end. Fields you leave out keep the theme default (see the
`*_stroke` defaults on `Theme`, such as `theme.budget_stroke`):

```python
from econ_viz import ArrowStyle, Stroke

canvas = Canvas(axis_stroke=Stroke(width=1.4, arrow=ArrowStyle.SIMPLE))
canvas.add_budget(2, 3, 30, stroke=Stroke(width=3, style="dashed"))
canvas.add_equilibrium(eq, drop_stroke=Stroke(style="dashdot"))
canvas.add_ray(0.5, stroke=Stroke(arrow=ArrowStyle.TRIANGLE))
```

`add_utility`, `add_path`, `add_decomposition`, `DemandDiagram`, and
`EdgeworthBox` take one `*_stroke` argument per kind of line they draw.

Point markers work the same way with `Marker` (colour, size, and shape);
fields you leave out keep the theme default, such as `theme.eq_marker`:

```python
from econ_viz import Marker

canvas.add_equilibrium(eq, marker=Marker(shape="s", size=8))
canvas.add_point(12, 2, label="A", marker=Marker(color="black", shape="D"))
canvas.add_decomposition(dec, point_marker=Marker(shape="^"))
```

Point labels take a `Label` (text, position, offset, colour, size, and
visibility) wherever a plain string worked. A label follows its point's
`Marker` colour unless it sets its own:

```python
from econ_viz import Label

canvas.add_equilibrium(eq, label=Label(position="bottom-left", offset=8))
canvas.add_point(12, 2, label=Label(text="A", position="left", fontsize=14))
canvas.add_utility(u, levels=3, ic_label=Label(text="U={:.1f}", position="top"))
canvas.add_decomposition(dec, point_label=Label(visible=False))  # hide A, B, C
```

Shade the budget set with `fill=True`, or pass a `Fill` for a colour and
opacity of its own (default `theme.budget_fill`, coloured like the line):

```python
from econ_viz import Fill

canvas.add_budget(2, 3, 30, color="black", fill=Fill(color="lightgrey", alpha=0.4))
```

Set a font for one canvas or a whole multi-panel figure without touching
Matplotlib's global settings. Pass a family name, a generic family such as
`"serif"`, or a fallback list:

```python
from econ_viz import Figure, Layout

canvas = Canvas(font=["Times New Roman", "serif"], math_font="stix")
figure = Figure(Layout.SIDE_BY_SIDE, font="serif", math_font="stix")
```

`font` applies to titles, axis labels, annotations, curve labels, and legends.
Math text, including the default axis labels, uses `math_font`: `"stix"`
(Times-like), `"cm"` (Computer Modern), `"dejavuserif"`, `"dejavusans"`, or
`"stixsans"`. An unavailable font raises
`InvalidParameterError`. TikZ output uses the LaTeX document's fonts, so only
generic families are mapped (`serif` → `\rmfamily`, `monospace` → `\ttfamily`).

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
econ-viz --version
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

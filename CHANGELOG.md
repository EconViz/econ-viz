# CHANGELOG

<!-- version list -->

## v1.12.0 (2026-09-26)

### Features

- Add `highlight_level` to `Canvas.add_utility(...)` so one focal indifference curve can be emphasized while the remaining levels are drawn as a subdued secondary set

- Add `secondary_stroke` plus theme-level secondary curve colour, linewidth, and opacity controls

- Add numeric and textbook-style ordinal curve labels (`u_1`, `u_2`, ...) with tangent-aligned placement that stays inside the visible plotting area

- Add a before/after/ordinal example for the indifference-curve hierarchy workflow

### Changes

- Apply indifference-curve label styling consistently to focal and secondary labels, while preserving smooth, linear, kinked, and satiation preferences

## v1.11.0 (2026-09-26)

### Features

- Complete the theme system so Canvas, Figure, decomposition diagrams, and Edgeworth boxes consistently use theme-defined backgrounds, labels, strokes, markers, and auxiliary economic lines

- Add four purpose-built themes: `paper` for journal figures, `monochrome` for black-and-white printing, `presentation` for projected slides, and `dark` for dark interfaces

- Add theme-level `background_color` and `label_scale` controls, plus defaults for titles, boxes, paths, kinks, bliss points, contract curves, cores, endowments, prices, and Walrasian equilibria

### Changes

- Make CLI/config theme resolution recognize every built-in theme and preserve visual semantics across Canvas, Figure, DemandDiagram, and EdgeworthBox

## v1.10.1 (2026-09-26)

### Maintenance

- Add Ruff linting and formatting checks plus Mypy type checking to local contributor tooling and CI

- Publish `econ-viz` as a PEP 561 typed package with an included `py.typed` marker

- Standardize the remaining public exception messages in English

### Bug Fixes

- Import NumPy in `analysis.levels`, fixing the unresolved `np.ndarray` annotation

- Bind the Pareto-search weight in each Edgeworth objective closure, preventing every objective from reusing the final loop value

## v1.10.0 (2026-09-26)

### Features

- Add `Config`, which loads diagram settings from an `econ-viz.toml` file: section names map to Theme properties (`[stroke.budget]`, `[label.point]`, `[marker.equilibrium]`, `[fill.budget]`, `[legend]`), plus `[color]`, `[font]`, and a `base` theme; `Config.load(...).use()` makes it the default for later diagrams

- Add `econ-viz init`, which writes a commented settings template, and `econ-viz plot --config FILE`

### Changes

- `Canvas`, `Figure`, and `EdgeworthBox` take their theme and fonts from the active `Config` when none is passed; with no config the output is unchanged

- Add `tomli` as a dependency on Python 3.10

## v1.9.0 (2026-09-26)

### Features

- Draw the indifference curves of a price-effect decomposition by default: U0 through A, U1 through C, and the curve through B under Slutsky; `show_curves=False` restores the previous figure, and `curve_stroke` / `curve_label` restyle and label them

- Add the `Haagsma` utility model (Haagsma, 2012), whose good x is always inferior and Giffen at high enough income, with closed-form `demand()` and `is_giffen()`

- Add `Legend`, placed automatically where it covers the least of the diagram (moving outside the plot area when every corner is taken), or at a chosen inside corner or outside side

- Accept `Label` for every text element: axis labels, the origin, titles, effect labels, and Edgeworth box text

- Add `opacity` to `Stroke`, `Marker`, `Label`, `Legend`, `Effect`, and `Fill`

### Changes

- Existing decomposition figures now include their indifference curves; pass `show_curves=False` if you already draw them with `add_utility`

- Effect range arrows below the x-axis point one way (A to B, B to C), and a zero effect draws no range arrow

- `Fill.alpha` becomes shorthand for `Fill.opacity`

### Bug Fixes

- Solve utility maximisation to a much tighter tolerance, so comparative statics and Slutsky matrices are accurate

- Draw indifference curves at negative utility levels as solid lines

- Make `scripts/release.sh` work on macOS

## v1.8.0 (2026-09-26)

### Features

- Add `Effect` to set the colour, range-arrow height, and label (text and position) of each price-effect decomposition effect

- Add `Marker` for point colour, size, and shape across `Canvas`, `DemandDiagram`, and `EdgeworthBox`

- Add `Label` for point-label text, position (four sides and four corners), offset, colour, size, and visibility; labels follow their point's `Marker` colour, and the decomposition's A/B/C labels can be hidden

- Add `Fill` so the budget-set shading can have its own colour and opacity

- Add `Axis` to set one axis's label, label position, and stroke in `Canvas`, `Figure`, `DemandDiagram`, and `EdgeworthBox`

- Add theme defaults for markers, labels, and fills, such as `theme.eq_marker`, `theme.point_label`, and `theme.budget_fill`

### Changes

- Reduce the default equilibrium point size from `6` to `4`

- Document `Stroke` as the preferred way to style lines; the separate `color` / `linewidth` / `linestyle` arguments remain as shorthand

### Bug Fixes

- Draw wedge arrowheads at a fixed on-screen size instead of stretching them along the whole line

- Apply `stroke_a` / `stroke_b` in `EdgeworthBox.add_equilibrium_indifference` and `add_indifference_curves_from_equilibrium`

## v1.7.0 (2026-09-25)

### Features

- Add independent axis-label positions, line styles, and arrowhead styles to `Canvas` and `Figure`

- Add per-figure `font` and `math_font` controls without changing Matplotlib's global configuration

- Add `Stroke` for per-line width, style, colour, and arrowheads across `Canvas`, `Figure`, `DemandDiagram`, and `EdgeworthBox`

- Add `econ-viz --version`

### Bug Fixes

- Validate utility-model parameter domains, correct the asymmetric CES expansion path and Cobb-Douglas limit, and allow satiation optima to leave budget unspent

- Make contour levels robust around zero and negative utility values

- Use one-sided finite differences when comparative statics approach price, income, or subsistence boundaries

- Make every example script runnable from a clean checkout

- Require tests to pass before tagged releases publish and include complete package licence metadata

### Maintenance

- Migrate package management and builds to `uv`

- Centralize shared numerical, rendering, and logging defaults

- Test Python 3.10 through 3.13, enforce branch coverage, and smoke-test all runnable examples in CI

- Document and automate the release workflow

## v1.6.0 (2026-04-24)

### Features

- Adopt a color-blind-friendly default palette and expose reusable `themes.COLORBLIND_CYCLE_RGB` / `themes.COLORBLIND_CYCLE_HEX`

- Reduce default IC linewidth from `2.0` to `1.8` for improved visual balance

### Documentation

- Add explicit palette/citation references in README and backfill missing changelog entries for `v1.3.2` and `v1.5.0`

### Chores

- Move animation example from `examples/animation.py` to `examples/scripts/animation.py` so `examples/` root only contains `__init__.py` and `scripts/`

- Stop tracking generated outputs under `examples/output/**` and remove tracked `examples/output/tikz/main.tex`

### Tests

- Update TikZ axis-spine regression assertion to avoid hard-coded color-index coupling

## v1.5.0 (2026-04-21)

### Features

- Add pure TikZ export backend (`econ_viz.io.backend_tikz`) and `.tex` support in `Canvas.save(...)` / `Figure.save(...)`

- Add price-effect decomposition APIs:
  - `decompose_price_effect(...)` / `PriceEffectDecomposition`
  - `Canvas.add_decomposition(...)` for A/B/C bundles, budget lines, and substitution/income overlays

- Expose decomposition interfaces at package root (`econ_viz.__init__`) for direct imports

- Expand examples with dedicated scripts for decomposition, demand diagrams, PCC/ICC paths, multi-layout figures, and multi-case TikZ export

### Refactors

- Improve axis-label placement and visibility toggling behavior for shared-panel layouts

### Tests

- Add dedicated regression tests for TikZ rendering/export and decomposition flows (`tests/test_tikz_backend.py`, `tests/test_optimizer.py`, `tests/test_canvas.py`)

### Bug Fixes

- Make notebook Colab install flow restart-safe

## v1.4.0 (2026-04-09)

### Features

- Add `Animator` class (`econ_viz.animation`) for parameter-sweep GIF export via Pillow — no ffmpeg required (closes #44)

- Add `WidgetViewer` class (`econ_viz.interactive`) for Jupyter notebook slider controls via ipywidgets (closes #45)

- Add optional dependency extras: `pip install econ-viz[animation]`, `pip install econ-viz[interactive]`, `pip install econ-viz[all]`

- Expand `examples/animation.py` with parameter, price, income, and budget-only GIF sweeps across common utility families

- Improve GIF export reliability by compositing frames onto a solid background and using restore-to-background disposal

- Extend notebook and widget examples so parameters can be adjusted by slider or direct numeric input in live Jupyter sessions

### Tests

- Add test coverage for Animator (init validation, GIF output, Pillow guard, frame disposal)

- Add test coverage for WidgetViewer (init validation, slider builder, numeric inputs, dependency guards)

## v1.3.2 (2026-04-03)

### Bug Fixes

- Enforce dashed black defaults for Edgeworth contract curves and price lines

### Chores

- Update publish workflow to skip uploading versions that already exist on PyPI

## v1.3.1 (2026-04-03)

### Features

- Add modular `Edgeworth` internals with dedicated compute/state/plotter helpers (`edgeworth_compute`, `edgeworth_state`, `edgeworth_plotter`)

- Add model registry (`models.registry`) and route CLI model construction through the registry

### Refactors

- Decouple CLI argument resolution from process exit flow:
  - introduce `CliConfigError`
  - centralize stderr/exit handling in `cli.main`

- Introduce shared contour level policies under `contours.level_policies` and reuse them across analysis/components/consumer paths

- Introduce shared figure exporter (`io.exporter`) and unify save logic across `Canvas`, `Figure`, and `EdgeworthBox`

- Split Canvas rendering responsibilities with renderer modules (`canvas.renderers.*`) and low-level primitives (`canvas.primitives`)

- Convert package root exports in `econ_viz.__init__` to lazy loading while preserving the public API surface

### Documentation

- Expand playground notebook with a complete Edgeworth Box section and export examples

## v1.3.0 (2026-04-03)

### Features

- Add `EdgeworthBox` and `EquilibriumFocusConfig` APIs for two-consumer exchange diagrams

- Add contract-curve construction (`mrs` / Pareto fallback), core rendering, Walrasian equilibrium overlay, and equilibrium-focused indifference-curve rendering

- Add `examples/edgeworth_box.py` covering common utility-function combinations

- Add dedicated Edgeworth test coverage in `tests/test_edgeworth.py`

## v1.2.3 (2026-03-31)

### Features

- Extend `SlutskyMatrix` with symmetry, negative-semidefiniteness, and homogeneity checks, plus validation warnings for failed theoretical conditions

- Expand CLI model coverage to include `QuasiLinear`, `StoneGeary`, and `Translog`

## v1.2.2 (2026-03-31)

### Bug Fixes

- Smooth PCC/ICC paths by default, extend endpoints slightly, and hide path markers unless requested

- Separate PCC/ICC path colours from indifference-curve colours and widen goods-space padding in demand diagrams

- Add a dedicated PCC/ICC example generator and coverage for path defaults and rendering

## v1.2.0 (2026-03-30)

### Features

- Add multi-panel `Figure` layouts and `Layout` enum (closes #7)

- Add linked `DemandDiagram` for Marshallian demand teaching figures (closes #31)

- Add `PricePath` / `IncomePath` helpers and `Canvas.add_path()` for PCC/ICC plots (closes #5)

## v1.1.0 (2026-03-30)

### Features

- Add comparative_statics helper (closes #12) ([#29](https://github.com/EconViz/econ-viz/pull/29),
  [`063def0`](https://github.com/EconViz/econ-viz/commit/063def0ee9677ca6dcf1bce8926fa4a4f248a021))

- Add HomogeneityAnalyzer and ReturnsToScale to analysis submodule (closes #14)
  ([#29](https://github.com/EconViz/econ-viz/pull/29),
  [`063def0`](https://github.com/EconViz/econ-viz/commit/063def0ee9677ca6dcf1bce8926fa4a4f248a021))

- Add Translog model (#9) and legend/IC label support
  ([#29](https://github.com/EconViz/econ-viz/pull/29),
  [`063def0`](https://github.com/EconViz/econ-viz/commit/063def0ee9677ca6dcf1bce8926fa4a4f248a021))


## v1.0.2 (2026-03-29)

### Bug Fixes

- Prevent double-wrap of math axis labels (closes #2)
  ([`7ee20a2`](https://github.com/EconViz/econ-viz/commit/7ee20a2b79e64f399d65c161a9776259a7dde092))

- Remove numpy upper bound to prevent Colab environment conflicts
  ([`b2e977f`](https://github.com/EconViz/econ-viz/commit/b2e977fbc58423ff4e63fd9846d404d236fb2347))


## v1.0.1 (2026-03-29)

### Bug Fixes

- Pin numpy<2 to prevent ABI mismatch on Colab and local installs
  ([`127bc26`](https://github.com/EconViz/econ-viz/commit/127bc26a19cb1c25e16339838e9f88dd4a2e3f5f))

### Chores

- Relax python/numpy bounds, pin pytest to 8.x
  ([`8dda610`](https://github.com/EconViz/econ-viz/commit/8dda6100c8c8c085c9f2b7bb98f7842ca60aa6a1))

### Documentation

- Add Stone-Geary to README, update test count badge
  ([`112d6a6`](https://github.com/EconViz/econ-viz/commit/112d6a6f069c34a3252523a333d8ea988b5b458b))


## v1.0.0 (2026-03-28)

- Initial Release

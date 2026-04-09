# CHANGELOG

<!-- version list -->

## v1.4.0 (2026-04-09)

### Features

- Add `Animator` class (`econ_viz.animation`) for parameter-sweep GIF export via Pillow — no ffmpeg required (closes #44)

- Add `WidgetViewer` class (`econ_viz.interactive`) for Jupyter notebook slider controls via ipywidgets (closes #45)

- Add optional dependency extras: `pip install econ-viz[animation]`, `pip install econ-viz[interactive]`, `pip install econ-viz[all]`

- Add usage examples in `examples/animation.py` (price sweep + income expansion GIFs)

### Tests

- Add test coverage for Animator (init validation, GIF output, Pillow guard)

- Add test coverage for WidgetViewer (init validation, slider builder, dependency guards)

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

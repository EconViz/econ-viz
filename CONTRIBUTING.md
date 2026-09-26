# Contributing to econ-viz

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Setting up the development environment

```bash
git clone https://github.com/EconViz/econ-viz.git
cd econ-viz
uv sync --all-extras
```

Run the complete local quality gate:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy econ_viz
```

Tests are grouped by package domain under `tests/`. The default command
measures statement and branch coverage and enforces the project coverage floor.
Use `uv run ruff format .` to apply formatting before rerunning the check.

## How to contribute

- **Bug reports** — open an issue with a minimal reproducible example
- **Feature requests** — open an issue describing the use case
- **Pull requests** — fork the repo, create a branch, and open a PR against `main`

Please make sure all tests pass before submitting a PR.

## Before you start — check the Project board

All planned work is tracked on the **[econ-viz Roadmap](https://github.com/orgs/EconViz/projects/1)**.

Before picking up an issue:

1. Check the board to see what is already **In Progress** — don't duplicate work.
2. Find an unassigned issue in **Todo** that you want to tackle.
3. Leave a comment on the issue saying you're working on it, then move it to **In Progress**.

This keeps everyone from stepping on each other.

## Good first issues

Issues labelled [`good first issue`](https://github.com/EconViz/econ-viz/issues?q=label%3A%22good+first+issue%22) are a great starting point. They are self-contained and well-documented.

## Code style

- Python 3.10+
- Ruff enforces lint rules and formatting
- Mypy checks the `econ_viz` package
- Add tests for any new behaviour

## Releasing

Merging to `main` never publishes a release. PyPI publishing runs only when a `v*` tag is pushed.

1. Open the release PR:

   ```bash
   scripts/release.sh prepare X.Y.Z
   ```

2. On the release branch, add a `## vX.Y.Z (YYYY-MM-DD)` section to the top of `CHANGELOG.md`.
3. In [econ-viz-docs](https://github.com/EconViz/econ-viz-docs), add a row for the release to the changelog page in all three languages, leaving out documentation-only changes:
   - `docs/project/changelog.md`
   - `docs/zh-TW/project/changelog.md`
   - `docs/zh-CN/project/changelog.md`

   Also update any guides that cover features added in this release. Publish these docs changes only after the release is on PyPI.
4. Merge the release PR, then tag and publish:

   ```bash
   scripts/release.sh finalize X.Y.Z
   ```

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE).

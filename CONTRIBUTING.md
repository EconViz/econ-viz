# Contributing to econ-viz

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Setting up the development environment

```bash
git clone https://github.com/EconViz/econ-viz.git
cd econ-viz
pip install poetry
poetry install --with dev
```

Run the test suite:

```bash
poetry run pytest
```

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

- Python 3.12+
- Follow the existing style (no linter is enforced, but keep it clean)
- Add tests for any new behaviour

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE).

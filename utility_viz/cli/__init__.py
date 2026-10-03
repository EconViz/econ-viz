"""Command-line interface for utility-viz.

Sub-commands
------------
models
    List all supported utility models and their parameters.
plot
    Generate a microeconomics diagram and save or display it.

Entry point
-----------
The ``utility-viz`` command is registered in ``pyproject.toml`` and
delegates to :func:`~utility_viz.cli.main.main`.
"""

from utility_viz.cli.main import main

__all__ = ["main"]

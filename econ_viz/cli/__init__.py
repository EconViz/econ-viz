"""Command-line interface for econ-viz.

Sub-commands
------------
models
    List all supported utility models and their parameters.
plot
    Generate a microeconomics diagram and save or display it.

Entry point
-----------
The ``econ-viz`` command is registered in ``pyproject.toml`` and
delegates to :func:`~econ_viz.cli.main.main`.
"""

from .main import main

__all__ = ["main"]

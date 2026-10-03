"""``init`` sub-command — write a commented utility-viz.toml template."""

from __future__ import annotations

import argparse
from pathlib import Path

from utility_viz.cli.errors import CliConfigError


def cmd_init(args: argparse.Namespace) -> None:
    """Write the template to *args.path*, refusing to overwrite unless ``--force``."""
    from utility_viz.core.config.settings import template

    path = Path(args.path)
    if path.exists() and not args.force:
        raise CliConfigError(f"{path} already exists; pass --force to overwrite it")
    path.write_text(template(), encoding="utf-8")
    print(f"Wrote {path}")

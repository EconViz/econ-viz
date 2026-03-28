"""``help`` sub-command — show help for the CLI or a specific command."""

from __future__ import annotations

import argparse


def cmd_help(
    args: argparse.Namespace,
    root_parser: argparse.ArgumentParser,
    subparsers: dict[str, argparse.ArgumentParser],
) -> None:
    """Print help for the CLI or a named sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.  ``args.topic`` is the optional sub-command name.
    root_parser : argparse.ArgumentParser
        Top-level parser, used when no topic is given.
    subparsers : dict[str, argparse.ArgumentParser]
        Mapping of sub-command name → its parser.
    """
    topic = getattr(args, "topic", None)
    if not topic:
        root_parser.print_help()
        return

    sub = subparsers.get(topic)
    if sub is None:
        available = ", ".join(k for k in subparsers if k != "help")
        root_parser.error(f"unknown command '{topic}'. Available: {available}")

    sub.print_help()

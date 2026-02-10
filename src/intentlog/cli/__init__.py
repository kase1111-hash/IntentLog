"""
IntentLog Command Line Interface

This package provides the modular CLI implementation for IntentLog.

Modules:
- core: Core commands (init, commit, branch, log, search, audit, status, diff, merge, config)
- analytics: Analytics commands (export, analytics, metrics)
- crypto: Cryptographic commands (keys, chain)
- formalize: Formalization commands (formalize)
- utils: Shared utilities
"""

import argparse
import sys

from .core import register_core_commands


def create_parser():
    """Create and configure the argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog="intentlog",
        description="IntentLog: Version Control for Human Reasoning"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Core commands always available
    register_core_commands(subparsers)

    # Optional command groups — register if their dependencies are available.
    # This allows the CLI to work even when optional packages (cryptography,
    # LLM providers, etc.) are not installed.
    _optional_commands = [
        (".analytics", "register_analytics_commands"),
        (".crypto", "register_crypto_commands"),
        (".formalize", "register_formalize_commands"),
        (".completion", "register_completion_commands"),
    ]
    for module_name, func_name in _optional_commands:
        try:
            import importlib
            mod = importlib.import_module(module_name, __name__)
            getattr(mod, func_name)(subparsers)
        except Exception:
            pass  # Command group unavailable — skip silently

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the command
    args.func(args)


__all__ = [
    'main',
    'create_parser',
]

"""
IntentLog Command Line Interface

This package provides the modular CLI implementation for IntentLog.

Modules:
- core: Core commands (init, commit, branch, log, search, audit, status, diff, merge, config)
- mp02: MP-02 Protocol commands (observe, segment, receipt, ledger, verify)
- analytics: Analytics commands (export, analytics, metrics, sufficiency)
- crypto: Cryptographic commands (keys, chain)
- privacy: Privacy commands (privacy)
- formalize: Formalization commands (formalize)
- utils: Shared utilities
"""

import argparse
import sys

from .core import register_core_commands
from .mp02 import register_mp02_commands
from .analytics import register_analytics_commands
from .crypto import register_crypto_commands
from .privacy import register_privacy_commands
from .formalize import register_formalize_commands


def create_parser():
    """Create and configure the argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog="intentlog",
        description="IntentLog: Version Control for Human Reasoning"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register all command groups
    register_core_commands(subparsers)
    register_mp02_commands(subparsers)
    register_analytics_commands(subparsers)
    register_crypto_commands(subparsers)
    register_privacy_commands(subparsers)
    register_formalize_commands(subparsers)

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
    'register_core_commands',
    'register_mp02_commands',
    'register_analytics_commands',
    'register_crypto_commands',
    'register_privacy_commands',
    'register_formalize_commands',
]

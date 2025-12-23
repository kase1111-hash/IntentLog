"""
IntentLog Command Line Interface

Provides the main CLI for interacting with IntentLog.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from .core import IntentLog
from .audit import audit_logs, print_audit_results
from .storage import (
    IntentLogStorage,
    ProjectNotFoundError,
    ProjectExistsError,
    BranchNotFoundError,
    BranchExistsError,
    compute_intent_hash,
)


def cmd_init(args):
    """Initialize a new IntentLog project"""
    project_name = args.project_name
    force = getattr(args, 'force', False)

    storage = IntentLogStorage()

    try:
        config = storage.init_project(project_name, force=force)
        print(f"Initialized IntentLog project: {config.project_name}")
        print(f"  Location: {storage.intentlog_dir}")
        print(f"  Branch: {config.current_branch}")
        print("Ready to track intent. Use 'ilog commit <message>' to add your first intent.")
    except ProjectExistsError as e:
        print(f"Error: {e}")
        print("Use --force to reinitialize.")
        sys.exit(1)


def cmd_commit(args):
    """Create a new intent commit"""
    message = args.message
    attach = getattr(args, 'attach', False)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build metadata
    metadata = {}

    # Handle --attach flag
    if attach:
        files = storage.get_attached_files()
        if files:
            file_hashes = storage.get_file_hashes(files)
            metadata["attached_files"] = file_hashes
            print(f"Attaching {len(files)} files...")

    # Parse message: first line is name, rest is reasoning
    lines = message.strip().split('\n', 1)
    if len(lines) == 1:
        # Single line: use as both name and reasoning
        name = lines[0][:50]  # Truncate for name
        reasoning = lines[0]
    else:
        name = lines[0]
        reasoning = lines[1].strip() if lines[1].strip() else lines[0]

    try:
        intent = storage.add_intent(
            name=name,
            reasoning=reasoning,
            metadata=metadata if metadata else None,
        )
        intent_hash = compute_intent_hash(intent)
        print(f"[{intent_hash}] {config.current_branch}: {name}")
        if attach and metadata.get("attached_files"):
            print(f"  {len(metadata['attached_files'])} files attached")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_branch(args):
    """Create or switch branches"""
    branch_name = args.branch_name
    list_branches = getattr(args, 'list', False)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if list_branches or branch_name is None:
        # List all branches
        branches = storage.list_branches()
        for branch in branches:
            marker = "*" if branch == config.current_branch else " "
            print(f"  {marker} {branch}")
        return

    # Check if branch exists
    existing_branches = storage.list_branches()

    if branch_name in existing_branches:
        # Switch to existing branch
        storage.switch_branch(branch_name)
        print(f"Switched to branch '{branch_name}'")
    else:
        # Create new branch
        try:
            storage.create_branch(branch_name)
            storage.switch_branch(branch_name)
            print(f"Created and switched to branch '{branch_name}'")
        except BranchExistsError as e:
            print(f"Error: {e}")
            sys.exit(1)


def cmd_log(args):
    """Show intent log history"""
    limit = getattr(args, 'limit', 10)
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    print(f"Intent log for '{branch}' ({len(intents)} total):\n")

    # Show most recent first, limited
    shown = 0
    for intent in reversed(intents):
        if shown >= limit:
            remaining = len(intents) - shown
            print(f"\n... and {remaining} more. Use --limit to see more.")
            break

        intent_hash = compute_intent_hash(intent)
        timestamp = intent.timestamp
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")

        print(f"[{intent_hash}] {timestamp}")
        print(f"  {intent.intent_name}")

        # Truncate long reasoning
        reasoning = intent.intent_reasoning
        if len(reasoning) > 100:
            reasoning = reasoning[:97] + "..."
        print(f"  {reasoning}")

        # Show attached files if any
        if intent.metadata.get("attached_files"):
            file_count = len(intent.metadata["attached_files"])
            print(f"  ({file_count} files attached)")

        print()
        shown += 1


def cmd_search(args):
    """Search intent history"""
    query = args.query
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        results = storage.search_intents(query, branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not results:
        print(f"No intents matching '{query}' on branch '{branch}'")
        return

    print(f"Found {len(results)} intent(s) matching '{query}':\n")

    for intent in results:
        intent_hash = compute_intent_hash(intent)
        timestamp = intent.timestamp
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")

        print(f"[{intent_hash}] {timestamp}")
        print(f"  {intent.intent_name}")

        # Highlight matching content in reasoning
        reasoning = intent.intent_reasoning
        if len(reasoning) > 150:
            # Try to show context around match
            query_lower = query.lower()
            idx = reasoning.lower().find(query_lower)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(reasoning), idx + len(query) + 50)
                reasoning = "..." + reasoning[start:end] + "..."
            else:
                reasoning = reasoning[:147] + "..."
        print(f"  {reasoning}")
        print()


def cmd_audit(args):
    """Audit intent logs"""
    log_file = args.log_file
    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)

    passed, errors = audit_logs(log_file)
    print_audit_results(passed, errors)

    sys.exit(0 if passed else 1)


def cmd_status(args):
    """Show project status"""
    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        intents = storage.load_intents()
        branches = storage.list_branches()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Project: {config.project_name}")
    print(f"Branch:  {config.current_branch}")
    print(f"Intents: {len(intents)}")
    print(f"Branches: {len(branches)} ({', '.join(branches)})")


def main():
    """Main CLI entry point"""
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

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new IntentLog project")
    init_parser.add_argument("project_name", help="Name of the project")
    init_parser.add_argument("--force", "-f", action="store_true", help="Force reinitialize")
    init_parser.set_defaults(func=cmd_init)

    # commit command
    commit_parser = subparsers.add_parser("commit", help="Create an intent commit")
    commit_parser.add_argument("message", help="Intent reasoning message")
    commit_parser.add_argument("--attach", "-a", action="store_true",
                               help="Attach git-tracked files to this commit")
    commit_parser.set_defaults(func=cmd_commit)

    # branch command
    branch_parser = subparsers.add_parser("branch", help="Create or switch branches")
    branch_parser.add_argument("branch_name", nargs="?", help="Name of the branch")
    branch_parser.add_argument("--list", "-l", action="store_true", help="List all branches")
    branch_parser.set_defaults(func=cmd_branch)

    # log command
    log_parser = subparsers.add_parser("log", help="Show intent history")
    log_parser.add_argument("--limit", "-n", type=int, default=10,
                            help="Number of intents to show (default: 10)")
    log_parser.add_argument("--branch", "-b", help="Show log for specific branch")
    log_parser.set_defaults(func=cmd_log)

    # search command
    search_parser = subparsers.add_parser("search", help="Search intent history")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--branch", "-b", help="Search in specific branch")
    search_parser.set_defaults(func=cmd_search)

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Audit intent logs")
    audit_parser.add_argument("log_file", help="Path to log file to audit")
    audit_parser.set_defaults(func=cmd_audit)

    # status command
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()

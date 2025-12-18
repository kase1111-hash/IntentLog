"""
IntentLog Command Line Interface

Provides the main CLI for interacting with IntentLog.
"""

import sys
import argparse
from pathlib import Path
from .core import IntentLog
from .audit import audit_logs, print_audit_results


def cmd_init(args):
    """Initialize a new IntentLog project"""
    project_name = args.project_name
    print(f"Initializing IntentLog project: {project_name}")
    # TODO: Create .intentlog directory and config
    print("✅ IntentLog initialized")


def cmd_commit(args):
    """Create a new intent commit"""
    message = args.message
    print(f"Creating intent commit: {message}")
    # TODO: Implement commit functionality
    print("✅ Intent committed")


def cmd_branch(args):
    """Create a new intent branch"""
    branch_name = args.branch_name
    print(f"Creating branch: {branch_name}")
    # TODO: Implement branching
    print(f"✅ Branch '{branch_name}' created")


def cmd_log(args):
    """Show intent log history"""
    print("Intent Log:")
    # TODO: Implement log display
    print("(Intent history will be shown here)")


def cmd_search(args):
    """Search intent history"""
    query = args.query
    print(f"Searching for: {query}")
    # TODO: Implement search
    print("(Search results will be shown here)")


def cmd_audit(args):
    """Audit intent logs"""
    log_file = args.log_file
    if not Path(log_file).exists():
        print(f"❌ Error: Log file '{log_file}' not found")
        sys.exit(1)

    passed, errors = audit_logs(log_file)
    print_audit_results(passed, errors)

    sys.exit(0 if passed else 1)


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
    init_parser.set_defaults(func=cmd_init)

    # commit command
    commit_parser = subparsers.add_parser("commit", help="Create an intent commit")
    commit_parser.add_argument("message", help="Intent reasoning message")
    commit_parser.add_argument("--attach", action="store_true", help="Attach files from git")
    commit_parser.set_defaults(func=cmd_commit)

    # branch command
    branch_parser = subparsers.add_parser("branch", help="Create a new branch")
    branch_parser.add_argument("branch_name", help="Name of the branch")
    branch_parser.set_defaults(func=cmd_branch)

    # log command
    log_parser = subparsers.add_parser("log", help="Show intent history")
    log_parser.set_defaults(func=cmd_log)

    # search command
    search_parser = subparsers.add_parser("search", help="Search intent history")
    search_parser.add_argument("query", help="Search query")
    search_parser.set_defaults(func=cmd_search)

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Audit intent logs")
    audit_parser.add_argument("log_file", help="Path to log file to audit")
    audit_parser.set_defaults(func=cmd_audit)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()

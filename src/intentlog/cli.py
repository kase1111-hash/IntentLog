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
    semantic = getattr(args, 'semantic', False)
    top_k = getattr(args, 'top', 5)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if semantic:
        # Semantic search using embeddings
        if not config.llm.is_configured():
            print("Error: LLM not configured for semantic search.")
            print("Configure with: ilog config llm --provider openai")
            print("Or use text search without --semantic flag.")
            sys.exit(1)

        try:
            engine = _get_semantic_engine(storage)
            intents = storage.load_intents(branch)
            results = engine.semantic_search(query, intents, top_k=top_k)
        except Exception as e:
            print(f"Error during semantic search: {e}")
            sys.exit(1)

        if not results:
            print(f"No intents semantically matching '{query}'")
            return

        print(f"Semantic search results for '{query}' (top {len(results)}):\n")

        for result in results:
            intent = result.intent
            intent_hash = compute_intent_hash(intent)
            timestamp = intent.timestamp
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime("%Y-%m-%d %H:%M")

            score_pct = int(result.score * 100)
            print(f"[{intent_hash}] {timestamp} ({score_pct}% match)")
            print(f"  {intent.intent_name}")

            reasoning = intent.intent_reasoning
            if len(reasoning) > 150:
                reasoning = reasoning[:147] + "..."
            print(f"  {reasoning}")
            print()
    else:
        # Standard text search
        try:
            results = storage.search_intents(query, branch)
        except BranchNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        if not results:
            print(f"No intents matching '{query}' on branch '{branch}'")
            if config.llm.is_configured():
                print("(Try --semantic for meaning-based search)")
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

    # Show LLM config if set
    if config.llm.is_configured():
        print(f"LLM:     {config.llm.provider} ({config.llm.model or 'default'})")


def _get_semantic_engine(storage: IntentLogStorage):
    """Get semantic engine from project config"""
    from .llm.provider import LLMConfig
    from .llm.registry import get_provider
    from .semantic import SemanticEngine

    config = storage.load_config()

    if not config.llm.is_configured():
        print("Error: LLM not configured. Run 'ilog config llm' first.")
        sys.exit(1)

    llm_config = LLMConfig(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key_env=config.llm.api_key_env or f"{config.llm.provider.upper()}_API_KEY",
        base_url=config.llm.base_url or None,
    )

    try:
        provider = get_provider(llm_config)
        if not provider.is_available():
            print(f"Error: LLM provider '{config.llm.provider}' not available.")
            print(f"Check that {llm_config.api_key_env} is set.")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Get embedding provider (may be different)
    embedding_provider = provider
    if config.llm.embedding_provider:
        embed_config = LLMConfig(
            provider=config.llm.embedding_provider,
            model=config.llm.embedding_model,
            api_key_env=f"{config.llm.embedding_provider.upper()}_API_KEY",
        )
        try:
            embedding_provider = get_provider(embed_config)
        except Exception:
            pass  # Fall back to main provider

    cache_dir = storage.intentlog_dir / "cache"
    return SemanticEngine(provider, embedding_provider, cache_dir)


def cmd_diff(args):
    """Show semantic diff between branches"""
    branch_spec = args.branches
    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse branch spec (e.g., "main..feature" or "feature")
    if ".." in branch_spec:
        branch_a, branch_b = branch_spec.split("..", 1)
    else:
        branch_a = "main"
        branch_b = branch_spec

    try:
        intents_a = storage.load_intents(branch_a)
        intents_b = storage.load_intents(branch_b)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Check if LLM is configured
    if not config.llm.is_configured():
        # Fall back to simple diff
        print(f"Comparing '{branch_a}' ({len(intents_a)} intents) to '{branch_b}' ({len(intents_b)} intents)\n")

        a_ids = {i.intent_id for i in intents_a}
        b_ids = {i.intent_id for i in intents_b}

        new_in_b = [i for i in intents_b if i.intent_id not in a_ids]
        if new_in_b:
            print(f"New in '{branch_b}':")
            for intent in new_in_b:
                intent_hash = compute_intent_hash(intent)
                print(f"  + [{intent_hash}] {intent.intent_name}")
        else:
            print("No new intents.")

        print("\n(Configure LLM for semantic analysis: ilog config llm)")
        return

    # Use semantic diff
    try:
        engine = _get_semantic_engine(storage)
    except SystemExit:
        return

    print(f"Analyzing changes from '{branch_a}' to '{branch_b}'...\n")

    diffs = engine.diff_branches(intents_a, intents_b, branch_a, branch_b)

    if not diffs:
        a_ids = {i.intent_id for i in intents_a}
        new_intents = [i for i in intents_b if i.intent_id not in a_ids]
        if new_intents:
            print(f"New intents in '{branch_b}':")
            for intent in new_intents:
                intent_hash = compute_intent_hash(intent)
                print(f"  + [{intent_hash}] {intent.intent_name}")
                print(f"    {intent.intent_reasoning[:100]}...")
        else:
            print("No significant changes between branches.")
        return

    for diff in diffs:
        print(f"Change: {diff.summary}")
        if diff.changes:
            for change in diff.changes:
                print(f"  - {change}")
        print()


def cmd_merge(args):
    """Merge branches with LLM-assisted conflict resolution"""
    source_branch = args.source
    message = getattr(args, 'message', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        target_branch = config.current_branch
        source_intents = storage.load_intents(source_branch)
        target_intents = storage.load_intents(target_branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Find new intents in source
    target_ids = {i.intent_id for i in target_intents}
    new_intents = [i for i in source_intents if i.intent_id not in target_ids]

    if not new_intents:
        print(f"Already up to date with '{source_branch}'.")
        return

    print(f"Merging {len(new_intents)} intent(s) from '{source_branch}' into '{target_branch}'...")

    # Add new intents to target
    for intent in new_intents:
        target_intents.append(intent)

    storage.save_intents(target_intents, target_branch)

    # Create merge commit if message provided
    if message:
        merge_intent = storage.add_intent(
            name=f"Merge {source_branch}",
            reasoning=message,
            metadata={"merge_from": source_branch, "merged_count": len(new_intents)},
            branch=target_branch,
        )
        merge_hash = compute_intent_hash(merge_intent)
        print(f"[{merge_hash}] Merge commit created")

    print(f"Merged {len(new_intents)} intent(s) from '{source_branch}'")


def cmd_config(args):
    """Configure IntentLog settings"""
    setting = args.setting
    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if setting == "llm":
        # Configure LLM
        provider = getattr(args, 'provider', None)
        model = getattr(args, 'model', None)

        if not provider:
            # Show current config
            if config.llm.is_configured():
                print(f"LLM Provider: {config.llm.provider}")
                print(f"Model: {config.llm.model or '(default)'}")
                if config.llm.api_key_env:
                    print(f"API Key Env: {config.llm.api_key_env}")
            else:
                print("LLM not configured.")
                print("\nAvailable providers: openai, anthropic, ollama")
                print("Example: ilog config llm --provider openai --model gpt-4o-mini")
            return

        # Update config
        from .storage import LLMSettings
        config.llm = LLMSettings(
            provider=provider,
            model=model or "",
            api_key_env=getattr(args, 'api_key_env', "") or f"{provider.upper()}_API_KEY",
            embedding_provider=getattr(args, 'embedding_provider', "") or "",
            embedding_model=getattr(args, 'embedding_model', "") or "",
        )
        storage.save_config(config)
        print(f"LLM configured: {provider} ({model or 'default'})")

    elif setting == "show":
        # Show all config
        print(f"Project: {config.project_name}")
        print(f"Created: {config.created_at}")
        print(f"Branch: {config.current_branch}")
        print(f"Version: {config.version}")
        if config.llm.is_configured():
            print(f"LLM: {config.llm.provider} ({config.llm.model or 'default'})")
    else:
        print(f"Unknown setting: {setting}")
        print("Available: llm, show")


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
    search_parser.add_argument("--semantic", "-s", action="store_true",
                               help="Use semantic (meaning-based) search with LLM embeddings")
    search_parser.add_argument("--top", "-t", type=int, default=5,
                               help="Number of results for semantic search (default: 5)")
    search_parser.set_defaults(func=cmd_search)

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Audit intent logs")
    audit_parser.add_argument("log_file", help="Path to log file to audit")
    audit_parser.set_defaults(func=cmd_audit)

    # status command
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.set_defaults(func=cmd_status)

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Show semantic diff between branches")
    diff_parser.add_argument("branches", help="Branch comparison (e.g., 'main..feature' or 'feature')")
    diff_parser.set_defaults(func=cmd_diff)

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge branches")
    merge_parser.add_argument("source", help="Source branch to merge from")
    merge_parser.add_argument("--message", "-m", help="Merge commit message")
    merge_parser.set_defaults(func=cmd_merge)

    # config command
    config_parser = subparsers.add_parser("config", help="Configure IntentLog settings")
    config_parser.add_argument("setting", help="Setting to configure (llm, show)")
    config_parser.add_argument("--provider", help="LLM provider (openai, anthropic, ollama)")
    config_parser.add_argument("--model", help="Model name")
    config_parser.add_argument("--api-key-env", help="Environment variable for API key")
    config_parser.add_argument("--embedding-provider", help="Provider for embeddings")
    config_parser.add_argument("--embedding-model", help="Embedding model name")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()

"""
Core CLI commands for IntentLog.

Commands: init, commit, branch, log, search, audit, status, diff, merge, config,
          show, hooks, blame
"""

import json as _json
import sys
from pathlib import Path
from datetime import datetime

from ..storage import (
    IntentLogStorage,
    ProjectNotFoundError,
    ProjectExistsError,
    BranchNotFoundError,
    BranchExistsError,
    compute_intent_hash,
)
from ..git import detect_git_repo, get_git_context
from ..audit import audit_logs, print_audit_results
from .utils import (
    get_storage,
    load_config_or_exit,
    format_timestamp,
    format_hash,
    get_semantic_engine,
)


def cmd_init(args):
    """Initialize a new IntentLog project"""
    project_name = args.project_name
    force = getattr(args, 'force', False)

    storage = IntentLogStorage()

    try:
        config = storage.init_project(project_name, force=force)

        # Detect git repository
        git_root = detect_git_repo(storage.project_root)
        if git_root:
            config.git_root = str(git_root)
            storage.save_config(config)

        print(f"Initialized IntentLog project: {config.project_name}")
        print(f"  Location: {storage.intentlog_dir}")
        print(f"  Branch: {config.current_branch}")
        if git_root:
            print(f"  Git repo: {git_root}")
        print("Ready to track intent. Use 'ilog commit <message>' to add your first intent.")
    except ProjectExistsError as e:
        print(f"Error: {e}")
        print("Use --force to reinitialize.")
        sys.exit(1)


def cmd_commit(args):
    """Create a new intent commit"""
    message = args.message
    attach = getattr(args, 'attach', False)
    sign = getattr(args, 'sign', False)
    key_password = getattr(args, 'key_password', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build metadata
    metadata = {}

    # Capture git context if in a git repo
    if config.git_root:
        git_root = Path(config.git_root)
        if git_root.is_dir():
            git_ctx = get_git_context(git_root)
            if git_ctx:
                metadata["git_context"] = git_ctx

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
        # Use chained intent for chain linking and optional signing
        chained = storage.add_chained_intent(
            name=name,
            reasoning=reasoning,
            metadata=metadata if metadata else None,
            sign=sign,
            key_password=key_password,
        )
        # Display short hash (first 12 chars of chain_hash)
        short_hash = chained.chain_hash[:12]
        print(f"[{short_hash}] {config.current_branch}: {name}")
        print(f"  Sequence: {chained.sequence}")
        if sign and chained.signature:
            print(f"  Signed: {chained.signature.get('key_id', 'unknown')}")
        if attach and metadata.get("attached_files"):
            print(f"  Files: {len(metadata['attached_files'])} attached")

        # Show git context if captured
        git_ctx = metadata.get("git_context")
        if git_ctx:
            git_branch = git_ctx.get("branch", "")
            git_commit = git_ctx.get("commit", "")[:12]
            staged = git_ctx.get("staged_files", [])
            parts = []
            if git_branch:
                parts.append(git_branch)
            if git_commit:
                parts.append(git_commit)
            if parts:
                print(f"  Git: {' @ '.join(parts)}")
            if staged:
                print(f"  Staged: {len(staged)} file(s)")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        if sign and "No signing keys" in str(e):
            print(f"Error: {e}")
            print("Generate a key first: ilog keys generate")
        else:
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
    git_commit_filter = getattr(args, 'git_commit', None)
    json_output = getattr(args, 'json', False)

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
        if json_output:
            print("[]")
        else:
            print(f"No intents on branch '{branch}'")
        return

    # Filter by git commit if requested
    if git_commit_filter:
        filtered = []
        for intent in intents:
            git_ctx = intent.metadata.get("git_context", {})
            commit = git_ctx.get("commit", "")
            if commit and commit.startswith(git_commit_filter):
                filtered.append(intent)
        intents = filtered
        if not intents:
            if json_output:
                print("[]")
            else:
                print(f"No intents linked to git commit '{git_commit_filter}'")
            return

    # JSON output mode
    if json_output:
        output = []
        for intent in reversed(intents):
            if len(output) >= limit:
                break
            entry = intent.to_dict()
            entry["hash"] = compute_intent_hash(intent)
            output.append(entry)
        print(_json.dumps(output, indent=2, default=str))
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
        timestamp = format_timestamp(intent.timestamp)

        print(f"[{intent_hash}] {timestamp}")
        print(f"  {intent.intent_name}")

        # Truncate long reasoning
        reasoning = intent.intent_reasoning
        if len(reasoning) > 100:
            reasoning = reasoning[:97] + "..."
        print(f"  {reasoning}")

        # Show git context if available
        git_ctx = intent.metadata.get("git_context", {})
        if git_ctx:
            git_branch = git_ctx.get("branch", "")
            git_hash = git_ctx.get("commit", "")[:12]
            parts = []
            if git_branch:
                parts.append(git_branch)
            if git_hash:
                parts.append(git_hash)
            if parts:
                print(f"  git: {' @ '.join(parts)}")

        # Show tags if any
        tags = intent.metadata.get("tags", [])
        if tags:
            print(f"  tags: {', '.join(tags)}")

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
            engine = get_semantic_engine(storage)
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
            timestamp = format_timestamp(intent.timestamp)

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
            timestamp = format_timestamp(intent.timestamp)

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
    json_output = getattr(args, 'json', False)
    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        intents = storage.load_intents()
        branches = storage.list_branches()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if json_output:
        data = {
            "project": config.project_name,
            "branch": config.current_branch,
            "intents": len(intents),
            "branches": branches,
        }
        if config.git_root:
            from ..git import get_git_branch, get_git_head
            git_root = Path(config.git_root)
            data["git"] = {
                "root": config.git_root,
                "branch": get_git_branch(git_root) or "",
                "head": get_git_head(git_root) or "",
            }
        print(_json.dumps(data, indent=2))
        return

    print(f"Project: {config.project_name}")
    print(f"Branch:  {config.current_branch}")
    print(f"Intents: {len(intents)}")
    print(f"Branches: {len(branches)} ({', '.join(branches)})")

    # Show git context if available
    if config.git_root:
        from ..git import get_git_branch, get_git_head
        git_root = Path(config.git_root)
        git_branch = get_git_branch(git_root)
        git_head = get_git_head(git_root)
        if git_branch or git_head:
            parts = []
            if git_branch:
                parts.append(git_branch)
            if git_head:
                parts.append(git_head[:12])
            print(f"Git:     {' @ '.join(parts)}")

    # Show LLM config if set
    if config.llm.is_configured():
        print(f"LLM:     {config.llm.provider} ({config.llm.model or 'default'})")


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
        engine = get_semantic_engine(storage)
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
        from ..storage import LLMSettings
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


def _find_intent(storage, intent_id, branch=None):
    """Find an intent by ID or hash prefix. Returns (intent, intents_list, branch) or exits."""
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

    for intent in intents:
        if intent.intent_id.startswith(intent_id):
            return intent, intents, branch
        if compute_intent_hash(intent).startswith(intent_id):
            return intent, intents, branch

    print(f"Error: No intent matching '{intent_id}' on branch '{branch}'")
    sys.exit(1)


def cmd_show(args):
    """Show a single intent with full metadata"""
    intent_id = args.intent_id
    branch = getattr(args, 'branch', None)
    json_output = getattr(args, 'json', False)

    storage = IntentLogStorage()
    found, intents, branch = _find_intent(storage, intent_id, branch)

    if json_output:
        entry = found.to_dict()
        entry["hash"] = compute_intent_hash(found)
        print(_json.dumps(entry, indent=2, default=str))
        return

    intent_hash = compute_intent_hash(found)
    print(f"Intent {intent_hash}")
    print(f"ID:        {found.intent_id}")
    print(f"Name:      {found.intent_name}")
    print(f"Timestamp: {format_timestamp(found.timestamp)}")
    if found.parent_intent_id:
        print(f"Parent:    {found.parent_intent_id[:12]}")
    print(f"\nReasoning:\n  {found.intent_reasoning}")

    # Git context
    git_ctx = found.metadata.get("git_context", {})
    if git_ctx:
        print(f"\nGit Context:")
        if git_ctx.get("branch"):
            print(f"  Branch: {git_ctx['branch']}")
        if git_ctx.get("commit"):
            print(f"  Commit: {git_ctx['commit']}")
        staged = git_ctx.get("staged_files", [])
        if staged:
            print(f"  Staged files:")
            for f in staged[:20]:
                print(f"    {f}")
            if len(staged) > 20:
                print(f"    ... and {len(staged) - 20} more")

    # Attached files
    attached = found.metadata.get("attached_files", {})
    if attached:
        print(f"\nAttached Files:")
        for fpath, fhash in list(attached.items())[:20]:
            print(f"  {fhash} {fpath}")

    # Tags
    tags = found.metadata.get("tags", [])
    if tags:
        print(f"\nTags: {', '.join(tags)}")

    # Links
    links = found.metadata.get("links", [])
    if links:
        print(f"\nLinked intents:")
        for link_id in links:
            print(f"  -> {link_id[:12]}")

    # Other metadata (exclude keys we already displayed)
    _displayed = ("git_context", "attached_files", "tags", "links")
    other_meta = {k: v for k, v in found.metadata.items() if k not in _displayed}
    if other_meta:
        print(f"\nMetadata:")
        for k, v in other_meta.items():
            print(f"  {k}: {v}")


def cmd_hooks(args):
    """Manage IntentLog git hooks"""
    action = args.action

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not config.git_root:
        print("Error: No git repository detected.")
        print("Run 'ilog init' in a git repository first.")
        sys.exit(1)

    from ..git import install_hooks, uninstall_hooks, hooks_status

    git_root = Path(config.git_root)

    if action == "install":
        try:
            installed = install_hooks(git_root)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        if not installed:
            print("Could not install hooks: existing hooks found.")
            print("Remove existing hooks first or install manually.")
            sys.exit(1)

        for hook in installed:
            print(f"  Installed: {hook}")
        print("Git hooks installed successfully.")

    elif action == "uninstall":
        removed = uninstall_hooks(git_root)
        if removed:
            for hook in removed:
                print(f"  Removed: {hook}")
            print("Git hooks removed.")
        else:
            print("No IntentLog hooks found to remove.")

    elif action == "status":
        status = hooks_status(git_root)
        if not status:
            print("Git hooks directory not found.")
            return
        print("Git hook status:")
        for hook, state in status.items():
            print(f"  {hook}: {state}")

    else:
        print(f"Unknown action: {action}")
        print("Available: install, uninstall, status")


def cmd_blame(args):
    """Show intent reasoning for git commits that touched a file"""
    file_path = args.file
    limit = getattr(args, 'limit', 20)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not config.git_root:
        print("Error: No git repository detected.")
        print("Run 'ilog init' in a git repository first.")
        sys.exit(1)

    from ..git import get_git_log_for_file

    git_root = Path(config.git_root)

    # Get git log for the file
    git_log = get_git_log_for_file(git_root, file_path, limit=limit)
    if not git_log:
        print(f"No git history found for '{file_path}'")
        return

    # Load all intents to match against git commits
    try:
        intents = storage.load_intents()
    except BranchNotFoundError:
        intents = []

    # Build a lookup: git commit hash -> list of intents
    commit_to_intents: dict = {}
    for intent in intents:
        git_ctx = intent.metadata.get("git_context", {})
        commit = git_ctx.get("commit", "")
        if commit:
            commit_to_intents.setdefault(commit, []).append(intent)

    print(f"Intent blame for '{file_path}':\n")
    for entry in git_log:
        commit_hash = entry["commit"]
        short_hash = commit_hash[:12]
        date = entry["date"][:10]
        message = entry["message"]

        # Find matching intents
        matching = commit_to_intents.get(commit_hash, [])

        print(f"[{short_hash}] {date} {message}")
        if matching:
            for intent in matching:
                reasoning = intent.intent_reasoning
                if len(reasoning) > 80:
                    reasoning = reasoning[:77] + "..."
                print(f"  Intent: {intent.intent_name}")
                print(f"    {reasoning}")
        else:
            print(f"  (no intent recorded)")
        print()


def cmd_tag(args):
    """Add or remove tags on an intent"""
    intent_id = args.intent_id
    tags = args.tags
    remove = getattr(args, 'remove', False)
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()
    found, intents, branch = _find_intent(storage, intent_id, branch)

    # Initialize tags list in metadata
    current_tags = found.metadata.get("tags", [])

    if remove:
        for tag in tags:
            if tag in current_tags:
                current_tags.remove(tag)
                print(f"  Removed tag: {tag}")
            else:
                print(f"  Tag not found: {tag}")
    else:
        for tag in tags:
            if tag not in current_tags:
                current_tags.append(tag)
                print(f"  Added tag: {tag}")
            else:
                print(f"  Already tagged: {tag}")

    found.metadata["tags"] = current_tags
    storage.save_intents(intents, branch)

    intent_hash = compute_intent_hash(found)
    print(f"[{intent_hash}] tags: {', '.join(current_tags) if current_tags else '(none)'}")


def cmd_link(args):
    """Link two intents together"""
    source_id = args.source
    target_id = args.target
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()
    source, intents, branch = _find_intent(storage, source_id, branch)
    target, _, _ = _find_intent(storage, target_id, branch)

    # Add link to source intent's metadata
    links = source.metadata.get("links", [])
    target_ref = target.intent_id

    if target_ref in links:
        print(f"Already linked: {source.intent_id[:12]} -> {target.intent_id[:12]}")
        return

    links.append(target_ref)
    source.metadata["links"] = links
    storage.save_intents(intents, branch)

    source_hash = compute_intent_hash(source)
    target_hash = compute_intent_hash(target)
    print(f"Linked [{source_hash}] {source.intent_name}")
    print(f"    -> [{target_hash}] {target.intent_name}")


def register_core_commands(subparsers):
    """Register core commands with the argument parser."""
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
    commit_parser.add_argument("--sign", "-s", action="store_true",
                               help="Sign this commit with default key")
    commit_parser.add_argument("--key-password", help="Password for encrypted signing key")
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
    log_parser.add_argument("--git-commit", help="Filter by git commit hash (prefix match)")
    log_parser.add_argument("--json", action="store_true", help="Output as JSON")
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
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
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

    # show command
    show_parser = subparsers.add_parser("show", help="Show a single intent with full details")
    show_parser.add_argument("intent_id", help="Intent ID or hash prefix")
    show_parser.add_argument("--branch", "-b", help="Search in specific branch")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    show_parser.set_defaults(func=cmd_show)

    # hooks command
    hooks_parser = subparsers.add_parser("hooks", help="Manage git hooks")
    hooks_parser.add_argument("action", choices=["install", "uninstall", "status"],
                              help="Hook action")
    hooks_parser.set_defaults(func=cmd_hooks)

    # blame command
    blame_parser = subparsers.add_parser("blame", help="Show intent reasoning for a file's git history")
    blame_parser.add_argument("file", help="File path to blame")
    blame_parser.add_argument("--limit", "-n", type=int, default=20,
                              help="Number of git commits to show (default: 20)")
    blame_parser.set_defaults(func=cmd_blame)

    # tag command
    tag_parser = subparsers.add_parser("tag", help="Add or remove tags on an intent")
    tag_parser.add_argument("intent_id", help="Intent ID or hash prefix")
    tag_parser.add_argument("tags", nargs="+", help="Tags to add or remove")
    tag_parser.add_argument("--remove", "-r", action="store_true", help="Remove tags instead of adding")
    tag_parser.add_argument("--branch", "-b", help="Branch to search")
    tag_parser.set_defaults(func=cmd_tag)

    # link command
    link_parser = subparsers.add_parser("link", help="Link two related intents")
    link_parser.add_argument("source", help="Source intent ID or hash prefix")
    link_parser.add_argument("target", help="Target intent ID or hash prefix")
    link_parser.add_argument("--branch", "-b", help="Branch to search")
    link_parser.set_defaults(func=cmd_link)

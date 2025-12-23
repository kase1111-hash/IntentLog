"""
IntentLog Command Line Interface

Provides the main CLI for interacting with IntentLog.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from .core import IntentLog, Intent
from .audit import audit_logs, print_audit_results
from .export import IntentExporter, ExportFilter, ExportFormat, AnonymizationConfig
from .analytics import IntentAnalytics, generate_summary
from .metrics import IntentMetrics
from .sufficiency import SufficiencyTest, run_sufficiency_test
from .semantic import FormalizationType, FormalizedOutput, SemanticEngine
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


# MP-02 Protocol Commands

def cmd_observe(args):
    """Start or manage observation session"""
    action = args.action
    paths = getattr(args, 'paths', [])

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .mp02.observer import TextObserver, CommandObserver, AnnotationObserver, ObserverConfig
    from .mp02.segmentation import SegmentationEngine, SegmentationRule
    from .mp02.ledger import Ledger

    ledger_dir = storage.intentlog_dir / "ledger"
    signals_dir = storage.intentlog_dir / "signals"
    signals_dir.mkdir(exist_ok=True)

    if action == "start":
        # Start observation session
        observer_config = ObserverConfig(capture_modality="file_watcher")

        # Create observers
        text_observer = TextObserver(paths=paths or ["."], config=observer_config)
        text_observer.start()

        print(f"Observation started for: {', '.join(paths or ['.'])}")
        print(f"Observer ID: {observer_config.observer_id}")
        print("Use 'ilog observe stop' to end session and generate receipts.")
        print("Use 'ilog segment mark' to add milestone markers.")

        # Save session info
        session_file = storage.intentlog_dir / "observe_session.json"
        import json
        with open(session_file, "w") as f:
            json.dump({
                "observer_id": observer_config.observer_id,
                "paths": paths or ["."],
                "started": datetime.now().isoformat(),
            }, f)

    elif action == "stop":
        # Stop observation and generate receipts
        session_file = storage.intentlog_dir / "observe_session.json"
        if not session_file.exists():
            print("No active observation session.")
            return

        print("Stopping observation session...")

        # Load signals from session
        import json
        signals_file = signals_dir / "pending_signals.json"
        if signals_file.exists():
            with open(signals_file, "r") as f:
                signals_data = json.load(f)
            print(f"Found {len(signals_data)} signals to process.")
        else:
            print("No signals captured in this session.")
            signals_data = []

        # Clean up session
        session_file.unlink()
        print("Session ended. Use 'ilog receipt create' to generate receipts from signals.")

    elif action == "status":
        # Show observation status
        session_file = storage.intentlog_dir / "observe_session.json"
        if session_file.exists():
            import json
            with open(session_file, "r") as f:
                session = json.load(f)
            print(f"Active session: {session['observer_id']}")
            print(f"Started: {session['started']}")
            print(f"Watching: {', '.join(session['paths'])}")
        else:
            print("No active observation session.")
            print("Start with: ilog observe start [paths...]")

    else:
        print(f"Unknown action: {action}")
        print("Available: start, stop, status")


def cmd_segment(args):
    """Mark segment boundaries"""
    action = args.action
    text = getattr(args, 'text', '')
    category = getattr(args, 'category', 'milestone')

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .mp02.observer import AnnotationObserver
    from .mp02.signal import Signal, SignalType

    signals_dir = storage.intentlog_dir / "signals"
    signals_dir.mkdir(exist_ok=True)

    if action == "mark":
        # Add a segment marker
        if not text:
            print("Error: Marker text required")
            print("Usage: ilog segment mark 'Completed initial design'")
            sys.exit(1)

        signal = Signal(
            signal_type=SignalType.ANNOTATION,
            timestamp=datetime.now(),
            content=text,
            metadata={"category": category},
        )

        # Append to pending signals
        import json
        signals_file = signals_dir / "pending_signals.json"
        signals = []
        if signals_file.exists():
            with open(signals_file, "r") as f:
                signals = json.load(f)

        signals.append(signal.to_dict())
        with open(signals_file, "w") as f:
            json.dump(signals, f, indent=2)

        print(f"[{category}] {text}")
        print(f"Signal ID: {signal.signal_id[:8]}")

    elif action == "list":
        # List pending segments
        import json
        signals_file = signals_dir / "pending_signals.json"
        if not signals_file.exists():
            print("No pending signals.")
            return

        with open(signals_file, "r") as f:
            signals = json.load(f)

        annotations = [s for s in signals if s.get("signal_type") == "annotation"]
        if not annotations:
            print("No segment markers found.")
            return

        print(f"Segment markers ({len(annotations)}):\n")
        for sig in annotations:
            ts = sig.get("timestamp", "")[:19]
            cat = sig.get("metadata", {}).get("category", "note")
            content = sig.get("content", "")[:60]
            print(f"  [{cat}] {ts} - {content}")

    else:
        print(f"Unknown action: {action}")
        print("Available: mark, list")


def cmd_receipt(args):
    """Generate or view effort receipts"""
    action = args.action
    receipt_id = getattr(args, 'receipt_id', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .mp02.signal import Signal
    from .mp02.segmentation import SegmentationEngine, SegmentationRule, EffortSegment
    from .mp02.validator import Validator
    from .mp02.receipt import ReceiptBuilder, verify_receipt
    from .mp02.ledger import Ledger

    receipts_dir = storage.intentlog_dir / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    ledger = Ledger(storage.intentlog_dir / "ledger")

    if action == "create":
        # Create receipts from pending signals
        import json
        signals_file = storage.intentlog_dir / "signals" / "pending_signals.json"

        if not signals_file.exists():
            print("No pending signals to process.")
            print("Use 'ilog observe start' to begin capturing signals.")
            return

        with open(signals_file, "r") as f:
            signals_data = json.load(f)

        if not signals_data:
            print("No signals to process.")
            return

        # Convert to Signal objects
        signals = [Signal.from_dict(s) for s in signals_data]
        print(f"Processing {len(signals)} signals...")

        # Segment signals
        rule = SegmentationRule(time_window_minutes=30)
        engine = SegmentationEngine(rule)
        segments = engine.segment(signals)

        if not segments:
            # Create single segment from all signals
            segments = [EffortSegment(signals=signals, rule=rule)]

        print(f"Created {len(segments)} segment(s)")

        # Create validator (rule-based without LLM for now)
        validator = Validator()

        # Generate receipts
        created = 0
        for segment in segments:
            validation = validator.validate(segment)
            builder = ReceiptBuilder()
            receipt = (builder
                .from_segment(segment)
                .with_validation(validation)
                .build())

            # Save receipt
            receipt_file = receipts_dir / f"{receipt.receipt_id[:8]}.json"
            with open(receipt_file, "w") as f:
                f.write(receipt.to_json())

            # Anchor to ledger
            entry = ledger.append(receipt)

            print(f"  [{receipt.receipt_hash[:8]}] {len(segment.signals)} signals, confidence: {validation.confidence:.2f}")
            created += 1

        # Clear pending signals
        signals_file.unlink()

        print(f"\nCreated {created} receipt(s). Use 'ilog receipt list' to view.")

    elif action == "list":
        # List receipts
        receipts = list(receipts_dir.glob("*.json"))
        if not receipts:
            print("No receipts found.")
            print("Use 'ilog receipt create' to generate receipts from signals.")
            return

        import json
        print(f"Effort receipts ({len(receipts)}):\n")
        for rfile in sorted(receipts, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            with open(rfile, "r") as f:
                data = json.load(f)
            receipt_hash = data.get("receipt_hash", "")[:8]
            start = data.get("start_time", "")[:19]
            signals = len(data.get("signal_hashes", []))
            confidence = data.get("confidence", 0)
            print(f"  [{receipt_hash}] {start} - {signals} signals (confidence: {confidence:.2f})")

    elif action == "show":
        # Show specific receipt
        if not receipt_id:
            print("Error: Receipt ID required")
            print("Usage: ilog receipt show <receipt_id>")
            sys.exit(1)

        # Find receipt file
        import json
        receipt_file = None
        for rfile in receipts_dir.glob("*.json"):
            if rfile.stem.startswith(receipt_id):
                receipt_file = rfile
                break

        if not receipt_file:
            print(f"Receipt not found: {receipt_id}")
            return

        with open(receipt_file, "r") as f:
            data = json.load(f)

        print(f"Receipt: {data['receipt_id']}")
        print(f"Hash: {data.get('receipt_hash', 'N/A')}")
        print(f"Time: {data.get('start_time')} to {data.get('end_time')}")
        print(f"Signals: {len(data.get('signal_hashes', []))}")
        print(f"Confidence: {data.get('confidence', 0):.2f}")
        print(f"\nSummary: {data.get('summary', 'N/A')}")

        if data.get("validation_metadata"):
            meta = data["validation_metadata"]
            print(f"\nValidator: {meta.get('model_name')} ({meta.get('provider')})")

    elif action == "verify":
        # Verify a receipt
        if not receipt_id:
            print("Error: Receipt ID required")
            sys.exit(1)

        import json
        from .mp02.receipt import Receipt

        receipt_file = None
        for rfile in receipts_dir.glob("*.json"):
            if rfile.stem.startswith(receipt_id):
                receipt_file = rfile
                break

        if not receipt_file:
            print(f"Receipt not found: {receipt_id}")
            return

        with open(receipt_file, "r") as f:
            receipt = Receipt.from_dict(json.load(f))

        report = verify_receipt(receipt)

        print(f"Verification: {'PASSED' if report['verified'] else 'FAILED'}")
        print(f"\nChecks passed:")
        for check in report['checks']:
            print(f"  + {check}")
        if report['warnings']:
            print(f"\nWarnings:")
            for warn in report['warnings']:
                print(f"  ! {warn}")
        if report['errors']:
            print(f"\nErrors:")
            for err in report['errors']:
                print(f"  x {err}")

    else:
        print(f"Unknown action: {action}")
        print("Available: create, list, show, verify")


def cmd_ledger(args):
    """Manage the append-only ledger"""
    action = args.action

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .mp02.ledger import Ledger, AnchoringService

    ledger = Ledger(storage.intentlog_dir / "ledger")

    if action == "show":
        # Show recent ledger entries
        limit = getattr(args, 'limit', 10)
        entries = ledger.read_entries(limit=limit)

        if not entries:
            print("Ledger is empty.")
            return

        print(f"Ledger entries (showing {len(entries)}):\n")
        for entry in entries:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            receipt = entry.receipt_hash[:8] if entry.receipt_hash else "(failure)"
            print(f"  [{entry.sequence}] {ts} - {receipt}")
            if entry.metadata.get("type") == "failure":
                print(f"      Failure: {entry.metadata.get('failure_type')}")

    elif action == "verify":
        # Verify ledger integrity
        print("Verifying ledger chain...")
        report = ledger.verify_chain()

        print(f"\nVerification: {'PASSED' if report['verified'] else 'FAILED'}")
        print(f"Entries checked: {report['entries_checked']}")

        if report['errors']:
            print("\nErrors:")
            for err in report['errors']:
                print(f"  x {err}")

        if report['warnings']:
            print("\nWarnings:")
            for warn in report['warnings']:
                print(f"  ! {warn}")

    elif action == "export":
        # Export ledger for verification
        output = getattr(args, 'output', None)
        if not output:
            output = "ledger_export.log"

        count = ledger.export(Path(output))
        print(f"Exported {count} entries to {output}")

    elif action == "stats":
        # Show ledger statistics
        stats = ledger.get_stats()

        print("Ledger Statistics:")
        print(f"  Path: {stats['ledger_path']}")
        print(f"  Entries: {stats['entry_count']}")
        print(f"  Size: {stats['file_size']} bytes")
        print(f"  Failures: {stats['failure_count']}")
        if stats['first_entry_time']:
            print(f"  First entry: {stats['first_entry_time']}")
        if stats['last_entry_time']:
            print(f"  Last entry: {stats['last_entry_time']}")

    elif action == "checkpoint":
        # Create a checkpoint
        service = AnchoringService(ledger)
        checkpoint = service.create_checkpoint()

        print(f"Checkpoint created: {checkpoint['checkpoint_id'][:8]}")
        print(f"Entries: {checkpoint['entry_count']}")
        print(f"Hash: {checkpoint['checkpoint_hash'][:16]}...")
        print(f"Chain verified: {checkpoint['chain_verified']}")

    else:
        print(f"Unknown action: {action}")
        print("Available: show, verify, export, stats, checkpoint")


def cmd_verify(args):
    """Verify integrity of receipts and ledger"""
    target = args.target

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .mp02.ledger import Ledger
    from .mp02.receipt import Receipt, verify_receipt

    ledger = Ledger(storage.intentlog_dir / "ledger")
    receipts_dir = storage.intentlog_dir / "receipts"

    if target == "all":
        # Verify everything
        print("Verifying IntentLog integrity...\n")

        # Verify ledger
        print("1. Ledger chain...")
        ledger_report = ledger.verify_chain()
        if ledger_report['verified']:
            print(f"   PASSED ({ledger_report['entries_checked']} entries)")
        else:
            print(f"   FAILED: {ledger_report['errors']}")

        # Verify receipts
        print("\n2. Receipts...")
        import json
        receipts = list(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
        passed = 0
        failed = 0
        for rfile in receipts:
            with open(rfile, "r") as f:
                receipt = Receipt.from_dict(json.load(f))
            report = verify_receipt(receipt)
            if report['verified']:
                passed += 1
            else:
                failed += 1
                print(f"   FAILED: {receipt.receipt_id[:8]}")

        print(f"   {passed} passed, {failed} failed")

        # Summary
        print("\n" + "="*40)
        overall = ledger_report['verified'] and failed == 0
        print(f"Overall: {'PASSED' if overall else 'FAILED'}")

    elif target == "ledger":
        # Verify ledger only
        report = ledger.verify_chain()
        print(f"Ledger: {'PASSED' if report['verified'] else 'FAILED'}")
        print(f"Entries: {report['entries_checked']}")

    elif target == "receipts":
        # Verify all receipts
        import json
        receipts = list(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
        for rfile in receipts:
            with open(rfile, "r") as f:
                receipt = Receipt.from_dict(json.load(f))
            report = verify_receipt(receipt)
            status = "PASSED" if report['verified'] else "FAILED"
            print(f"  [{receipt.receipt_id[:8]}] {status}")

    else:
        # Assume it's a receipt ID
        import json
        receipt_file = None
        for rfile in receipts_dir.glob("*.json"):
            if rfile.stem.startswith(target):
                receipt_file = rfile
                break

        if receipt_file:
            with open(receipt_file, "r") as f:
                receipt = Receipt.from_dict(json.load(f))
            report = verify_receipt(receipt)
            print(f"Receipt {target}: {'PASSED' if report['verified'] else 'FAILED'}")
        else:
            print(f"Unknown target: {target}")
            print("Available: all, ledger, receipts, or <receipt_id>")


# Phase 4: Analytics and Metrics Commands

def cmd_export(args):
    """Export intents for evaluation or fine-tuning"""
    format_type = getattr(args, 'format', 'jsonl')
    output = getattr(args, 'output', None)
    anonymize = getattr(args, 'anonymize', False)
    branch = getattr(args, 'branch', None)
    start_date = getattr(args, 'start', None)
    end_date = getattr(args, 'end', None)
    category = getattr(args, 'category', None)

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

    # Build filter config
    filter_config = ExportFilter()
    if start_date:
        filter_config.start_date = datetime.fromisoformat(start_date)
    if end_date:
        filter_config.end_date = datetime.fromisoformat(end_date)
    if category:
        filter_config.categories = [category]

    # Build format config
    format_config = ExportFormat(
        format_type=format_type,
        pretty_print=(format_type == 'json'),
    )

    # Build anonymization config
    anonymization = AnonymizationConfig() if anonymize else None

    # Create exporter
    exporter = IntentExporter(
        filter_config=filter_config,
        anonymization=anonymization,
        format_config=format_config,
    )

    # Export
    if output:
        output_path = Path(output)
        result = exporter.export(intents, output_path)
        stats = exporter.get_stats(intents)
        print(f"Exported {stats['filtered_intents']} intents to {output}")
        print(f"  Format: {format_type}")
        if anonymize:
            print(f"  Anonymized: Yes")
        if stats['filter_ratio'] < 1.0:
            print(f"  Filtered: {stats['total_intents']} -> {stats['filtered_intents']}")
    else:
        # Print to stdout
        result = exporter.export(intents)
        print(result)


def cmd_analytics(args):
    """Generate analytics report for intents"""
    action = args.action
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

    analytics = IntentAnalytics(intents)

    if action == "summary" or action is None:
        # Generate full summary
        summary = generate_summary(intents)
        print(summary)

    elif action == "latency":
        # Show latency statistics
        stats = analytics.compute_latency_stats()
        print("Latency Statistics:")
        print(f"  Count: {stats.count}")
        print(f"  Mean: {stats.mean:.2f}ms")
        print(f"  Median: {stats.median:.2f}ms")
        print(f"  Std Dev: {stats.std_dev:.2f}ms")
        print(f"  Min: {stats.min}ms")
        print(f"  Max: {stats.max}ms")
        print(f"  P95: {stats.p95:.2f}ms")
        print(f"  P99: {stats.p99:.2f}ms")

    elif action == "frequency":
        # Show frequency statistics
        stats = analytics.compute_frequency_stats()
        print("Frequency Statistics:")
        print(f"  Total intents: {stats.total_count}")
        print(f"  Date range: {stats.date_range_days} days")
        print(f"  Intents per day: {stats.intents_per_day:.2f}")
        print(f"  Intents per hour: {stats.intents_per_hour:.2f}")
        print(f"  Peak hour: {stats.peak_hour}:00")
        print(f"  Peak day: {stats.peak_day}")
        print("\nTop categories:")
        for cat, count in list(stats.by_category.items())[:5]:
            print(f"    {cat}: {count}")

    elif action == "errors":
        # Show error statistics
        stats = analytics.compute_error_stats()
        print("Error Statistics:")
        print(f"  Total errors: {stats.total_errors}")
        print(f"  Error rate: {stats.error_rate * 100:.2f}%")
        if stats.errors_by_type:
            print("\nBy type:")
            for typ, count in stats.errors_by_type.items():
                print(f"    {typ}: {count}")

    elif action == "trends":
        # Show trending intents
        window = getattr(args, 'window', 7)
        top_n = getattr(args, 'top', 10)
        trends = analytics.get_trending_intents(window_days=window, top_n=top_n)
        print(f"Trending Intents (last {window} days):\n")
        for intent_name, count in trends:
            print(f"  {count:3d} x {intent_name}")

    elif action == "bottlenecks":
        # Show bottlenecks
        threshold = getattr(args, 'threshold', None)
        top_n = getattr(args, 'top', 10)
        bottlenecks = analytics.get_bottlenecks(
            latency_threshold_ms=threshold,
            top_n=top_n
        )
        print("Bottlenecks (high latency intents):\n")
        for intent_name, avg_latency in bottlenecks:
            print(f"  {avg_latency:7.2f}ms - {intent_name}")

    elif action == "report":
        # Generate full report
        report = analytics.generate_report()
        print("="*60)
        print("INTENT ANALYTICS REPORT")
        print("="*60)
        print(f"\nDate Range: {report.date_range['start']} to {report.date_range['end']}")
        print(f"Total Intents: {report.total_intents}")
        print(f"\nLatency: mean={report.latency.mean:.2f}ms, p95={report.latency.p95:.2f}ms")
        print(f"Frequency: {report.frequency.intents_per_day:.2f}/day")
        print(f"Errors: {report.errors.error_rate * 100:.2f}%")
        if report.activity:
            print(f"Sessions: {report.activity.session_count}")
        print("="*60)

    else:
        print(f"Unknown action: {action}")
        print("Available: summary, latency, frequency, errors, trends, bottlenecks, report")


def cmd_metrics(args):
    """Compute doctrine metrics for intents"""
    action = args.action
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

    metrics = IntentMetrics(intents)

    if action == "all" or action is None:
        # Show all metrics
        all_metrics = metrics.get_all_metrics()
        print("="*60)
        print("DOCTRINE METRICS")
        print("="*60)

        density = all_metrics['intent_density']
        print(f"\nIntent Density (Di): {density['Di']:.3f}")
        print(f"  Resolution: {density['resolution']:.3f}")
        print(f"  Continuity: {density['continuity']:.3f}")
        print(f"  Coverage: {density['coverage']:.3f}")

        info = all_metrics['information_density']
        print(f"\nInformation Density:")
        print(f"  Avg words: {info['avg_words']:.1f}")
        print(f"  Avg chars: {info['avg_chars']:.1f}")
        print(f"  Compression ratio: {info['compression_ratio']:.3f}")

        audit = all_metrics['auditability']
        print(f"\nAuditability: {audit['score']:.3f} ({audit['rating']})")

        fraud = all_metrics['fraud_resistance']
        print(f"Fraud Resistance: {fraud['score']:.3f} ({fraud['rating']})")
        print("="*60)

    elif action == "density":
        # Show intent density
        density = metrics.compute_intent_density()
        print("Intent Density Metrics:")
        print(f"  Di: {density.Di:.3f}")
        print(f"  Resolution (R): {density.resolution:.3f}")
        print(f"  Continuity (C): {density.continuity:.3f}")
        print(f"  Coverage (Co): {density.coverage:.3f}")
        print(f"  Sample size: {density.sample_size}")

    elif action == "info":
        # Show information density
        info = metrics.compute_information_density()
        print("Information Density:")
        print(f"  Avg words: {info.avg_words:.1f}")
        print(f"  Avg chars: {info.avg_chars:.1f}")
        print(f"  Unique terms ratio: {info.unique_terms_ratio:.3f}")
        print(f"  Compression ratio: {info.compression_ratio:.3f}")
        print(f"  Entropy: {info.entropy:.3f}")

    elif action == "auditability":
        # Show auditability score
        audit = metrics.compute_auditability()
        print(f"Auditability Score: {audit.score:.3f}")
        print(f"Rating: {audit.rating}")
        print("\nComponents:")
        for comp, val in audit.components.items():
            print(f"  {comp}: {val:.3f}")

    elif action == "fraud":
        # Show fraud resistance
        fraud = metrics.compute_fraud_resistance()
        print(f"Fraud Resistance Score: {fraud.score:.3f}")
        print(f"Rating: {fraud.rating}")
        print("\nFactors:")
        for factor, val in fraud.factors.items():
            print(f"  {factor}: {val:.3f}")

    else:
        print(f"Unknown action: {action}")
        print("Available: all, density, info, auditability, fraud")


def cmd_sufficiency(args):
    """Run Intent Sufficiency Test"""
    branch = getattr(args, 'branch', None)
    author = getattr(args, 'author', None)
    verbose = getattr(args, 'verbose', False)

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

    # Run sufficiency test
    report = run_sufficiency_test(intents, expected_author=author)

    print("="*60)
    print("INTENT SUFFICIENCY TEST")
    print("="*60)
    print(f"\nResult: {'PASS' if report.passed else 'FAIL'}")
    print(f"Score: {report.overall_score:.2f}/5.00")
    print(f"Criteria passed: {report.criteria_passed}/{report.total_criteria}")

    print("\nCriteria Results:")
    for criterion, result in report.criteria.items():
        status = "PASS" if result.passed else "FAIL"
        symbol = "✓" if result.passed else "✗"
        print(f"  {symbol} {criterion}: {result.score:.2f} - {status}")
        if verbose and not result.passed:
            for issue in result.issues:
                print(f"      - {issue}")

    if report.recommendations and verbose:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")

    print("="*60)
    sys.exit(0 if report.passed else 1)


def cmd_formalize(args):
    """Derive formal code/rules/heuristics from prose intent"""
    action = args.action
    output_type = getattr(args, 'type', 'code')
    language = getattr(args, 'language', None)
    intent_id = getattr(args, 'intent_id', None)
    query = getattr(args, 'query', None)
    branch = getattr(args, 'branch', None)
    output_file = getattr(args, 'output', None)

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

    # Map string to FormalizationType
    type_map = {
        'code': FormalizationType.CODE,
        'rules': FormalizationType.RULES,
        'heuristics': FormalizationType.HEURISTICS,
        'schema': FormalizationType.SCHEMA,
        'config': FormalizationType.CONFIG,
        'spec': FormalizationType.SPEC,
        'tests': FormalizationType.TESTS,
    }
    formalization_type = type_map.get(output_type, FormalizationType.CODE)

    # Get LLM provider
    from .llm.registry import get_provider

    llm_config = config.llm_config or {}
    provider_name = llm_config.get("provider", "mock")

    try:
        provider = get_provider(provider_name, llm_config)
    except Exception as e:
        print(f"Error: Could not initialize LLM provider: {e}")
        print("Configure LLM with: ilog config llm --provider openai")
        sys.exit(1)

    engine = SemanticEngine(provider)

    try:
        if action == "intent":
            # Formalize a specific intent by ID
            if not intent_id:
                print("Error: --intent-id required for 'intent' action")
                sys.exit(1)

            # Find intent by ID (partial match)
            target_intent = None
            for intent in intents:
                if intent.intent_id.startswith(intent_id):
                    target_intent = intent
                    break

            if not target_intent:
                print(f"Error: No intent found matching ID '{intent_id}'")
                sys.exit(1)

            print(f"Formalizing intent: {target_intent.intent_name}")
            print(f"  ID: {target_intent.intent_id[:12]}...")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize(
                target_intent,
                formalization_type=formalization_type,
                language=language,
            )

        elif action == "chain":
            # Formalize from the entire chain of intents
            print(f"Formalizing chain of {len(intents)} intents")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize_chain(
                intents,
                formalization_type=formalization_type,
                language=language,
            )

        elif action == "search":
            # Formalize from search results
            if not query:
                print("Error: --query required for 'search' action")
                sys.exit(1)

            print(f"Searching for intents matching: '{query}'")
            print(f"  Type: {output_type.upper()}")
            if language:
                print(f"  Language: {language}")
            print()

            result = engine.formalize_from_search(
                query,
                intents,
                formalization_type=formalization_type,
                language=language,
            )

        else:
            print(f"Unknown action: {action}")
            sys.exit(1)

        # Display result
        print("=" * 60)
        print(f"FORMALIZED OUTPUT ({output_type.upper()})")
        print("=" * 60)

        if result.language:
            print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence:.0%}")
        print()
        print(result.content)
        print()

        if result.explanation:
            print("-" * 60)
            print("EXPLANATION:")
            print(result.explanation)
            print()

        if result.warnings:
            print("-" * 60)
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning}")
            print()

        print("-" * 60)
        print("PROVENANCE:")
        print(f"  Source intents: {len(result.provenance.source_intent_ids)}")
        for sid in result.provenance.source_intent_ids[:5]:
            print(f"    - {sid[:12]}...")
        if len(result.provenance.source_intent_ids) > 5:
            print(f"    ... and {len(result.provenance.source_intent_ids) - 5} more")
        print(f"  Formalized at: {result.provenance.formalized_at}")
        print(f"  Model: {result.provenance.model}")
        print("=" * 60)

        # Output to file if requested
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\nSaved to: {output_file}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during formalization: {e}")
        sys.exit(1)


def cmd_keys(args):
    """Manage cryptographic signing keys"""
    action = args.action

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from .crypto import KeyManager, CryptoNotAvailableError, KeyNotFoundError

    try:
        key_manager = KeyManager(storage.intentlog_dir)
    except CryptoNotAvailableError as e:
        print(f"Error: {e}")
        print("Install cryptography: pip install cryptography")
        sys.exit(1)

    if action == "generate":
        name = getattr(args, 'name', 'default')
        password = getattr(args, 'password', None)

        try:
            key_pair = key_manager.generate_key(name=name, password=password)
            print(f"Generated key pair: {name}")
            print(f"  Key ID: {key_pair.key_id}")
            print(f"  Algorithm: Ed25519")
            print(f"  Created: {key_pair.created_at}")
            if password:
                print(f"  Encrypted: Yes")
            print(f"\nPublic key saved to: .intentlog/keys/{name}.pub")
            print(f"Private key saved to: .intentlog/keys/{name}.key")
        except Exception as e:
            print(f"Error generating key: {e}")
            sys.exit(1)

    elif action == "list":
        keys = key_manager.list_keys()
        default_key = key_manager.get_default_key_name()

        if not keys:
            print("No keys configured.")
            print("Generate a key: ilog keys generate")
            return

        print("Signing keys:\n")
        for name, meta in keys.items():
            marker = "*" if name == default_key else " "
            encrypted = "(encrypted)" if meta.get("encrypted") else ""
            print(f"  {marker} {name}: {meta.get('key_id', 'unknown')} {encrypted}")
            print(f"      Created: {meta.get('created_at', 'unknown')}")

    elif action == "export":
        name = getattr(args, 'name', 'default') or key_manager.get_default_key_name()
        if not name:
            print("Error: No key specified and no default key set")
            sys.exit(1)

        try:
            public_pem = key_manager.export_public_key(name)
            output = getattr(args, 'output', None)

            if output:
                with open(output, 'w') as f:
                    f.write(public_pem)
                print(f"Public key exported to: {output}")
            else:
                print(public_pem)
        except KeyNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "default":
        name = getattr(args, 'name', None)
        if not name:
            default = key_manager.get_default_key_name()
            print(f"Default key: {default or '(none)'}")
        else:
            try:
                key_manager.set_default_key(name)
                print(f"Default key set to: {name}")
            except KeyNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

    else:
        print(f"Unknown action: {action}")
        print("Available: generate, list, export, default")


def cmd_chain(args):
    """Manage intent chain (Merkle tree)"""
    action = args.action
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if action == "verify":
        print(f"Verifying chain integrity for branch '{branch}'...\n")

        result = storage.verify_chain(branch)

        if result.valid:
            print(f"Chain verification: PASSED")
        else:
            print(f"Chain verification: FAILED")

        print(f"Entries checked: {result.entries_checked}")
        print(f"Root hash: {result.root_hash[:16]}...")

        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  x {err}")

        if result.warnings:
            print("\nWarnings:")
            for warn in result.warnings:
                print(f"  ! {warn}")

        if result.broken_at is not None:
            print(f"\nChain broken at sequence: {result.broken_at}")

        sys.exit(0 if result.valid else 1)

    elif action == "migrate":
        print(f"Migrating branch '{branch}' to chain format...")

        count = storage.migrate_to_chain(branch)

        if count > 0:
            print(f"Migrated {count} intents to chain format.")
            root_hash = storage.get_root_hash(branch)
            print(f"Root hash: {root_hash[:16]}...")
        else:
            print("No intents to migrate.")

    elif action == "status":
        try:
            chained = storage.load_chained_intents(branch)
        except BranchNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        if not chained:
            print(f"No intents on branch '{branch}'")
            return

        print(f"Chain status for '{branch}':\n")
        print(f"  Intents: {len(chained)}")
        print(f"  Root hash: {storage.get_root_hash(branch)[:16]}...")

        # Check for signatures
        signed_count = sum(1 for c in chained if c.signature)
        print(f"  Signed: {signed_count}/{len(chained)}")

        # Verify chain
        result = storage.verify_chain(branch)
        print(f"  Chain valid: {'Yes' if result.valid else 'No'}")

    elif action == "proof":
        sequence = getattr(args, 'sequence', None)
        if sequence is None:
            print("Error: --sequence required for proof")
            sys.exit(1)

        try:
            proof = storage.get_inclusion_proof(int(sequence), branch)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"Inclusion proof for sequence {sequence}:\n")
        print(f"  Intent ID: {proof['target_intent_id']}")
        print(f"  Intent hash: {proof['target_intent_hash'][:16]}...")
        print(f"  Root hash: {proof['root_hash'][:16]}...")
        print(f"  Chain length: {proof['chain_length']}")
        print(f"  Proof path length: {len(proof['proof_path'])}")

        # Verify the proof
        from .merkle import verify_inclusion_proof
        valid = verify_inclusion_proof(proof)
        print(f"\n  Proof valid: {'Yes' if valid else 'No'}")

    else:
        print(f"Unknown action: {action}")
        print("Available: verify, migrate, status, proof")


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

    # MP-02 Protocol Commands

    # observe command
    observe_parser = subparsers.add_parser("observe", help="Start or manage observation session (MP-02)")
    observe_parser.add_argument("action", choices=["start", "stop", "status"],
                                help="Observation action")
    observe_parser.add_argument("paths", nargs="*", help="Paths to observe (for start)")
    observe_parser.set_defaults(func=cmd_observe)

    # segment command
    segment_parser = subparsers.add_parser("segment", help="Mark segment boundaries (MP-02)")
    segment_parser.add_argument("action", choices=["mark", "list"],
                                help="Segment action")
    segment_parser.add_argument("text", nargs="?", help="Marker text (for mark)")
    segment_parser.add_argument("--category", "-c", default="milestone",
                                help="Marker category (milestone, note, decision, etc.)")
    segment_parser.set_defaults(func=cmd_segment)

    # receipt command
    receipt_parser = subparsers.add_parser("receipt", help="Generate or view effort receipts (MP-02)")
    receipt_parser.add_argument("action", choices=["create", "list", "show", "verify"],
                                help="Receipt action")
    receipt_parser.add_argument("receipt_id", nargs="?", help="Receipt ID (for show/verify)")
    receipt_parser.set_defaults(func=cmd_receipt)

    # ledger command
    ledger_parser = subparsers.add_parser("ledger", help="Manage append-only ledger (MP-02)")
    ledger_parser.add_argument("action", choices=["show", "verify", "export", "stats", "checkpoint"],
                               help="Ledger action")
    ledger_parser.add_argument("--limit", "-n", type=int, default=10,
                               help="Number of entries to show (default: 10)")
    ledger_parser.add_argument("--output", "-o", help="Output file (for export)")
    ledger_parser.set_defaults(func=cmd_ledger)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify integrity (MP-02)")
    verify_parser.add_argument("target", nargs="?", default="all",
                               help="What to verify: all, ledger, receipts, or <receipt_id>")
    verify_parser.set_defaults(func=cmd_verify)

    # Phase 4: Analytics and Metrics Commands

    # export command
    export_parser = subparsers.add_parser("export", help="Export intents for eval/fine-tuning")
    export_parser.add_argument("--format", "-f", default="jsonl",
                               choices=["json", "jsonl", "csv", "huggingface", "openai"],
                               help="Output format (default: jsonl)")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--anonymize", "-a", action="store_true",
                               help="Anonymize exported data")
    export_parser.add_argument("--branch", "-b", help="Export from specific branch")
    export_parser.add_argument("--start", help="Filter: start date (ISO format)")
    export_parser.add_argument("--end", help="Filter: end date (ISO format)")
    export_parser.add_argument("--category", "-c", help="Filter by category")
    export_parser.set_defaults(func=cmd_export)

    # analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Generate analytics reports")
    analytics_parser.add_argument("action", nargs="?", default="summary",
                                  choices=["summary", "latency", "frequency", "errors",
                                           "trends", "bottlenecks", "report"],
                                  help="Analytics action (default: summary)")
    analytics_parser.add_argument("--branch", "-b", help="Analyze specific branch")
    analytics_parser.add_argument("--window", "-w", type=int, default=7,
                                  help="Time window in days for trends (default: 7)")
    analytics_parser.add_argument("--top", "-t", type=int, default=10,
                                  help="Number of top results (default: 10)")
    analytics_parser.add_argument("--threshold", type=int,
                                  help="Latency threshold in ms for bottlenecks")
    analytics_parser.set_defaults(func=cmd_analytics)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Compute doctrine metrics")
    metrics_parser.add_argument("action", nargs="?", default="all",
                                choices=["all", "density", "info", "auditability", "fraud"],
                                help="Metrics to compute (default: all)")
    metrics_parser.add_argument("--branch", "-b", help="Analyze specific branch")
    metrics_parser.set_defaults(func=cmd_metrics)

    # sufficiency command
    sufficiency_parser = subparsers.add_parser("sufficiency", help="Run Intent Sufficiency Test")
    sufficiency_parser.add_argument("--branch", "-b", help="Test specific branch")
    sufficiency_parser.add_argument("--author", "-a", help="Expected author for attribution")
    sufficiency_parser.add_argument("--verbose", "-v", action="store_true",
                                    help="Show detailed issues and recommendations")
    sufficiency_parser.set_defaults(func=cmd_sufficiency)

    # formalize command (Deferred Formalization)
    formalize_parser = subparsers.add_parser("formalize", help="Derive code/rules/heuristics from prose intent")
    formalize_parser.add_argument("action", choices=["intent", "chain", "search"],
                                  help="Formalization action: intent (single), chain (all), search (by query)")
    formalize_parser.add_argument("--type", "-t", default="code",
                                  choices=["code", "rules", "heuristics", "schema", "config", "spec", "tests"],
                                  help="Output type (default: code)")
    formalize_parser.add_argument("--language", "-l",
                                  help="Programming language for code output (default: python)")
    formalize_parser.add_argument("--intent-id", "-i",
                                  help="Intent ID to formalize (for 'intent' action)")
    formalize_parser.add_argument("--query", "-q",
                                  help="Search query (for 'search' action)")
    formalize_parser.add_argument("--branch", "-b", help="Branch to formalize from")
    formalize_parser.add_argument("--output", "-o", help="Output file (saves full result as JSON)")
    formalize_parser.set_defaults(func=cmd_formalize)

    # Phase 2: Cryptographic Integrity Commands

    # keys command
    keys_parser = subparsers.add_parser("keys", help="Manage cryptographic signing keys")
    keys_parser.add_argument("action", choices=["generate", "list", "export", "default"],
                             help="Key management action")
    keys_parser.add_argument("--name", "-n", default="default",
                             help="Key name (default: 'default')")
    keys_parser.add_argument("--password", "-p", help="Password for key encryption")
    keys_parser.add_argument("--output", "-o", help="Output file for export")
    keys_parser.set_defaults(func=cmd_keys)

    # chain command
    chain_parser = subparsers.add_parser("chain", help="Manage intent hash chain")
    chain_parser.add_argument("action", choices=["verify", "migrate", "status", "proof"],
                              help="Chain action")
    chain_parser.add_argument("--branch", "-b", help="Branch to operate on")
    chain_parser.add_argument("--sequence", "-s", type=int,
                              help="Sequence number for proof")
    chain_parser.set_defaults(func=cmd_chain)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()

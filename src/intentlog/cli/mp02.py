"""
MP-02 Protocol CLI commands for IntentLog.

Commands: observe, segment, receipt, ledger, verify
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from ..storage import IntentLogStorage, ProjectNotFoundError
from .utils import load_config_or_exit


def cmd_observe(args):
    """Start or manage observation session"""
    action = args.action
    paths = getattr(args, 'paths', [])

    storage = IntentLogStorage()
    config = load_config_or_exit(storage)

    from ..mp02.observer import TextObserver, ObserverConfig

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
        signals_file = signals_dir / "pending_signals.json"
        if signals_file.exists():
            with open(signals_file, "r") as f:
                signals_data = json.load(f)
            print(f"Found {len(signals_data)} signals to process.")
        else:
            print("No signals captured in this session.")

        # Clean up session
        session_file.unlink()
        print("Session ended. Use 'ilog receipt create' to generate receipts from signals.")

    elif action == "status":
        # Show observation status
        session_file = storage.intentlog_dir / "observe_session.json"
        if session_file.exists():
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
    config = load_config_or_exit(storage)

    from ..mp02.signal import Signal, SignalType

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
    config = load_config_or_exit(storage)

    from ..mp02.signal import Signal
    from ..mp02.segmentation import SegmentationEngine, SegmentationRule, EffortSegment
    from ..mp02.validator import Validator
    from ..mp02.receipt import ReceiptBuilder, verify_receipt
    from ..mp02.ledger import Ledger

    receipts_dir = storage.intentlog_dir / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    ledger = Ledger(storage.intentlog_dir / "ledger")

    if action == "create":
        # Create receipts from pending signals
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
            ledger.append(receipt)

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

        from ..mp02.receipt import Receipt

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
    config = load_config_or_exit(storage)

    from ..mp02.ledger import Ledger, AnchoringService

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
    config = load_config_or_exit(storage)

    from ..mp02.ledger import Ledger
    from ..mp02.receipt import Receipt, verify_receipt

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
        receipts = list(receipts_dir.glob("*.json")) if receipts_dir.exists() else []
        for rfile in receipts:
            with open(rfile, "r") as f:
                receipt = Receipt.from_dict(json.load(f))
            report = verify_receipt(receipt)
            status = "PASSED" if report['verified'] else "FAILED"
            print(f"  [{receipt.receipt_id[:8]}] {status}")

    else:
        # Assume it's a receipt ID
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


def register_mp02_commands(subparsers):
    """Register MP-02 protocol commands with the argument parser."""
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

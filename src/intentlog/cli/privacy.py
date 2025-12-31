"""
Privacy CLI commands for IntentLog (MP-02 Section 12).

Commands: privacy
"""

import sys
from pathlib import Path

from ..storage import IntentLogStorage, ProjectNotFoundError
from .utils import load_config_or_exit


def cmd_privacy(args):
    """
    Manage privacy controls per MP-02 Section 12.

    - Raw signals MAY be encrypted or access-controlled
    - Receipts MUST NOT expose raw content by default
    - Humans MAY revoke future observation
    - Past receipts remain immutable
    """
    action = args.action
    storage = IntentLogStorage()
    config = load_config_or_exit(storage)

    from ..privacy import (
        PrivacyManager,
        ENCRYPTION_AVAILABLE,
    )

    privacy_manager = PrivacyManager(Path(storage.project_root))

    if action == "status":
        # Show privacy status
        print("Privacy Status (MP-02 Section 12):\n")

        # Encryption status
        print("Encryption:")
        if ENCRYPTION_AVAILABLE:
            keys = privacy_manager.key_manager.list_keys()
            print(f"  Available: Yes (cryptography library installed)")
            print(f"  Keys configured: {len(keys)}")
            if keys:
                default_key = privacy_manager.key_manager.get_default_key()
                if default_key:
                    print(f"  Default key: {default_key.key_id[:8]}...")
        else:
            print("  Available: No (install 'cryptography' package)")

        # Revocation status
        print("\nRevocation:")
        revocations = privacy_manager.revocation_manager.list_revocations()
        print(f"  Active revocations: {len(revocations)}")
        if privacy_manager.revocation_manager.is_revoked():
            print("  Status: OBSERVATION REVOKED")
        else:
            print("  Status: Active (observation allowed)")

        # Show recent revocations
        if revocations:
            print("\n  Recent revocations:")
            for rev in sorted(revocations, key=lambda r: r.revoked_at, reverse=True)[:3]:
                print(f"    [{rev.record_id[:8]}] {rev.target_type}: {rev.target_id or 'all'}")

    elif action == "revoke":
        # Revoke future observation (MP-02 Section 12)
        target = getattr(args, 'target', 'all')
        reason = getattr(args, 'reason', None)
        user_id = getattr(args, 'user_id', 'cli-user')

        if target == "all":
            record = privacy_manager.revoke_future_observation(user_id, reason)
            print("Future observation REVOKED (MP-02 Section 12)")
            print(f"  Record ID: {record.record_id}")
            print(f"  Scope: {record.scope}")
            print(f"  Revoked by: {record.revoked_by}")
            print("\nNote: Past receipts remain immutable per MP-02 specification.")
        elif target == "intent":
            intent_id = getattr(args, 'target_id', None)
            if not intent_id:
                print("Error: --target-id required for intent revocation")
                sys.exit(1)
            record = privacy_manager.revocation_manager.revoke_intent(intent_id, user_id, reason)
            print(f"Intent {intent_id} revoked for future access.")
        elif target == "session":
            session_id = getattr(args, 'target_id', None)
            if not session_id:
                print("Error: --target-id required for session revocation")
                sys.exit(1)
            record = privacy_manager.revocation_manager.revoke_session(session_id, user_id, reason)
            print(f"Session {session_id} revoked for future access.")
        else:
            print(f"Unknown target: {target}")
            print("Available: all, intent, session")

    elif action == "list":
        # List revocations
        revocations = privacy_manager.revocation_manager.list_revocations()

        if not revocations:
            print("No revocations on record.")
            return

        print(f"Revocation Records ({len(revocations)}):\n")
        for rev in sorted(revocations, key=lambda r: r.revoked_at, reverse=True):
            print(f"[{rev.record_id[:8]}] {rev.revoked_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Type: {rev.target_type}")
            if rev.target_id:
                print(f"  Target: {rev.target_id}")
            print(f"  By: {rev.revoked_by}")
            if rev.reason:
                print(f"  Reason: {rev.reason}")
            print()

    elif action == "encrypt":
        # Encrypt intents
        if not ENCRYPTION_AVAILABLE:
            print("Error: Encryption not available.")
            print("Install cryptography: pip install cryptography")
            sys.exit(1)

        branch = getattr(args, 'branch', None) or config.current_branch
        intents = storage.load_intents(branch)

        if not intents:
            print("No intents to encrypt.")
            return

        # Get or generate key
        key = privacy_manager.key_manager.get_default_key()
        if key is None:
            print("Generating encryption key...")
            key = privacy_manager.key_manager.generate_key(name="auto-generated")
            print(f"  Key ID: {key.key_id[:8]}...")

        # Encrypt intents
        encrypted_count = 0
        for intent in intents:
            if intent.get("_encryption_key_id"):
                continue  # Already encrypted

            privacy_manager.encrypt_intent(intent)
            encrypted_count += 1

        print(f"Encrypted {encrypted_count} intent(s) with key {key.key_id[:8]}...")
        print("\nNote: Raw signal content is protected. Receipts store only hashes.")

    elif action == "keys":
        # List encryption keys
        if not ENCRYPTION_AVAILABLE:
            print("Encryption not available. Install 'cryptography' package.")
            return

        keys = privacy_manager.key_manager.list_keys()

        if not keys:
            print("No encryption keys configured.")
            print("Use 'ilog privacy encrypt' to auto-generate a key.")
            return

        print(f"Encryption Keys ({len(keys)}):\n")
        for meta in keys:
            encrypted = "(encrypted)" if meta.get("encrypted") else ""
            default = " [default]" if meta.get("is_default") else ""
            print(f"  {meta.get('key_id', 'unknown')[:8]}...{default}")
            print(f"    Name: {meta.get('name', 'unnamed')}")
            print(f"    Created: {meta.get('created_at', 'unknown')[:19]}")
            if meta.get("revoked"):
                print("    Status: REVOKED")
            print()

    else:
        print(f"Unknown action: {action}")
        print("Available: status, revoke, list, encrypt, keys")


def register_privacy_commands(subparsers):
    """Register privacy commands with the argument parser."""
    # privacy command (MP-02 Section 12)
    privacy_parser = subparsers.add_parser("privacy", help="Manage privacy controls (MP-02 Section 12)")
    privacy_parser.add_argument("action", choices=["status", "revoke", "list", "encrypt", "keys"],
                                help="Privacy action: status, revoke, list revocations, encrypt intents, list keys")
    privacy_parser.add_argument("--target", "-t", default="all",
                                choices=["all", "intent", "session"],
                                help="Revocation target (default: all)")
    privacy_parser.add_argument("--target-id", help="ID of intent/session to revoke")
    privacy_parser.add_argument("--reason", "-r", help="Reason for revocation")
    privacy_parser.add_argument("--user-id", "-u", default="cli-user",
                                help="User ID performing action (default: cli-user)")
    privacy_parser.add_argument("--branch", "-b", help="Branch for encrypt action")
    privacy_parser.set_defaults(func=cmd_privacy)

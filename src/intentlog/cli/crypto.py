"""
Cryptographic CLI commands for IntentLog.

Commands: keys, chain
"""

import sys

from ..storage import IntentLogStorage, ProjectNotFoundError, BranchNotFoundError
from .utils import load_config_or_exit


def cmd_keys(args):
    """Manage cryptographic signing keys"""
    action = args.action

    storage = IntentLogStorage()
    config = load_config_or_exit(storage)

    from ..crypto import KeyManager, CryptoNotAvailableError, KeyNotFoundError

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
        from ..merkle import verify_inclusion_proof
        valid = verify_inclusion_proof(proof)
        print(f"\n  Proof valid: {'Yes' if valid else 'No'}")

    else:
        print(f"Unknown action: {action}")
        print("Available: verify, migrate, status, proof")


def register_crypto_commands(subparsers):
    """Register cryptographic commands with the argument parser."""
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

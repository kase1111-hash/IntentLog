"""
Basic IntentLog Usage Examples

This file demonstrates how to use IntentLog in your projects.
"""

from intentlog import IntentLog, Intent

def example_basic_logging():
    """Basic intent logging"""
    print("=== Basic Intent Logging ===\n")

    # Create an intent log
    log = IntentLog(project_name="my_project")

    # Add intents
    intent1 = log.add_intent(
        name="initialize_system",
        reasoning="Setting up the core system components to prepare for data processing"
    )
    print(f"Created intent: {intent1.intent_name}")

    intent2 = log.add_intent(
        name="load_configuration",
        reasoning="Loading configuration to ensure proper system setup before processing",
        parent_id=intent1.intent_id
    )
    print(f"Created nested intent: {intent2.intent_name}")

    # Export to see the structure
    export = log.export_to_dict()
    print(f"\nTotal intents logged: {len(export['intents'])}\n")


def example_search():
    """Searching through intent history"""
    print("=== Searching Intent History ===\n")

    log = IntentLog(project_name="search_demo")

    # Add multiple intents
    log.add_intent(
        name="setup_database",
        reasoning="Initialize database connection for persistent storage"
    )
    log.add_intent(
        name="optimize_query",
        reasoning="Optimize database query to reduce latency"
    )
    log.add_intent(
        name="cache_results",
        reasoning="Add caching layer to improve performance"
    )

    # Search for intents related to database
    results = log.search_intents("database")
    print(f"Found {len(results)} intents related to 'database':")
    for intent in results:
        print(f"  - {intent.intent_name}: {intent.intent_reasoning}")

    print()


def example_intent_chain():
    """Working with intent chains (parent-child relationships)"""
    print("=== Intent Chains ===\n")

    log = IntentLog(project_name="chain_demo")

    # Create a chain of intents
    root = log.add_intent(
        name="start_migration",
        reasoning="Beginning database migration to new schema"
    )

    step1 = log.add_intent(
        name="backup_data",
        reasoning="Create backup before migration to ensure data safety",
        parent_id=root.intent_id
    )

    step2 = log.add_intent(
        name="run_migrations",
        reasoning="Execute migration scripts to update schema",
        parent_id=step1.intent_id
    )

    step3 = log.add_intent(
        name="verify_integrity",
        reasoning="Verify data integrity after migration to ensure correctness",
        parent_id=step2.intent_id
    )

    # Get the full chain
    chain = log.get_intent_chain(step3.intent_id)
    print("Full intent chain:")
    for i, intent in enumerate(chain, 1):
        print(f"  {i}. {intent.intent_name}")

    print()


def example_persistent_storage():
    """Persisting intents to disk and verifying the hash chain"""
    print("=== Persistent Storage & Chain Verification ===\n")

    import tempfile
    from pathlib import Path

    from intentlog import IntentLogStorage

    # Use a throwaway project directory so this runs from any cwd
    with tempfile.TemporaryDirectory(prefix="intentlog_example_") as tmp:
        storage = IntentLogStorage(project_path=Path(tmp))
        storage.init_project("storage_demo")

        # Each chained intent is linked to the previous one by hash
        first = storage.add_chained_intent(
            name="choose_storage_format",
            reasoning="Using JSON files so the history stays human-readable and diffable",
        )
        second = storage.add_chained_intent(
            name="enable_chain_linking",
            reasoning="Linking each intent to the previous hash so tampering is detectable",
            parent_id=first.intent.intent_id,
        )
        print(f"Persisted {second.sequence + 1} intents to {storage.intentlog_dir}")

        # Search the stored history
        results = storage.search_intents("hash")
        print(f"Found {len(results)} stored intent(s) mentioning 'hash'")

        # Verify the chain that was written to disk
        verification = storage.verify_chain()
        status = "valid" if verification.valid else "BROKEN"
        print(f"Chain {status}: {verification.entries_checked} entries checked")
        print(f"Root hash: {verification.root_hash[:16]}...\n")


if __name__ == "__main__":
    example_basic_logging()
    example_search()
    example_intent_chain()
    example_persistent_storage()

    print("✅ All examples completed!")

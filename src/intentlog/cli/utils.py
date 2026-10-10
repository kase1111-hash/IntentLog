"""
Shared utilities for IntentLog CLI commands.
"""

import sys
from pathlib import Path
from datetime import datetime

from ..storage import IntentLogStorage, ProjectNotFoundError, compute_intent_hash


def get_storage():
    """Get IntentLogStorage instance."""
    return IntentLogStorage()


def load_config_or_exit(storage: IntentLogStorage):
    """Load config or exit with error message."""
    try:
        return storage.load_config()
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def format_timestamp(timestamp) -> str:
    """Format a timestamp for display."""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return str(timestamp)[:16]


def format_hash(hash_str: str, length: int = 12) -> str:
    """Format a hash for display (truncated)."""
    return hash_str[:length] if hash_str else "N/A"


def get_semantic_engine(storage: IntentLogStorage):
    """Get semantic engine from project config."""
    from ..llm.provider import LLMConfig
    from ..llm.registry import get_provider
    from ..semantic import SemanticEngine

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

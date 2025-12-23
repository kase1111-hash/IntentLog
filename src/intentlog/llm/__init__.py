"""
IntentLog LLM Integration Module

Provides pluggable LLM backends for semantic features like
diffs, search, and merge conflict resolution.
"""

from .provider import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
)
from .registry import get_provider, register_provider, list_providers

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "get_provider",
    "register_provider",
    "list_providers",
]

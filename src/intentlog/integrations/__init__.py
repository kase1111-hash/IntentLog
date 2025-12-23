"""
IntentLog Integrations

This module provides integration points with external systems like Memory Vault.
"""

from .memory_vault import (
    MemoryVaultIntegration,
    MemoryVaultConfig,
    ClassificationLevel,
)
from .llm_classifier import (
    LLMIntentClassifier,
    IntentCategory,
    ClassificationResult,
    classify_intent_with_llm,
)

__all__ = [
    # Memory Vault
    "MemoryVaultIntegration",
    "MemoryVaultConfig",
    "ClassificationLevel",
    # LLM Classification
    "LLMIntentClassifier",
    "IntentCategory",
    "ClassificationResult",
    "classify_intent_with_llm",
]

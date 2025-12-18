"""
Memory Vault Integration for IntentLog

This module provides integration between IntentLog and Memory Vault for
secure, classified storage of high-value intent records.

Based on Memory-Vault-Integration.md specification.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class ClassificationLevel(IntEnum):
    """Memory Vault classification levels for intents"""
    TRANSIENT = 0  # Low-value, synth-mind only
    LEARNED_HEURISTIC = 1  # Learned patterns
    FAILED_PATH_LESSON = 2  # Lessons from failures
    LONG_TERM_GOAL = 3  # Strategic goals
    RECOVERY_SEED = 5  # Critical recovery information


@dataclass
class MemoryVaultConfig:
    """Configuration for Memory Vault integration"""
    vault_path: str = ".memory_vault"
    encryption_profile: str = "default-passphrase"
    enable_tpm: bool = False
    default_classification: int = 2


class MemoryVaultIntegration:
    """
    Integration layer between IntentLog and Memory Vault.

    This class manages the dual persistence model:
    - Low-value intents → local memory
    - High-value intents → Memory Vault with classification gates
    """

    def __init__(self, config: Optional[MemoryVaultConfig] = None):
        self.config = config or MemoryVaultConfig()
        self._vault = None

    def _get_vault(self):
        """Lazy load Memory Vault if available"""
        if self._vault is None:
            try:
                from memory_vault.vault import MemoryVault
                self._vault = MemoryVault()
            except ImportError:
                raise ImportError(
                    "Memory Vault not installed. Install with: pip install memory-vault"
                )
        return self._vault

    def store_critical_intent(
        self,
        intent_id: str,
        content: bytes,
        classification: int,
        metadata: Optional[Dict[str, Any]] = None,
        cooldown_seconds: int = 0
    ) -> str:
        """
        Store a high-value intent artifact in the Memory Vault.

        Args:
            intent_id: The IntentLog intent ID
            content: The intent content as bytes
            classification: Classification level (1-5)
            metadata: Additional metadata
            cooldown_seconds: Cooldown period before recall

        Returns:
            Memory Vault memory_id for linking
        """
        try:
            from memory_vault.models import MemoryObject
            from uuid import uuid4

            vault = self._get_vault()

            obj = MemoryObject(
                memory_id=str(uuid4()),
                content_plaintext=content,
                classification=classification,
                encryption_profile=self.config.encryption_profile,
                intent_ref=intent_id,
                access_policy={"cooldown_seconds": cooldown_seconds},
                value_metadata=(metadata or {}) | {
                    "source": "IntentLog",
                    "intent_id": intent_id
                }
            )

            vault.store_memory(obj)
            return obj.memory_id

        except ImportError as e:
            raise ImportError(
                "Memory Vault not available. This is an optional integration. "
                f"Error: {e}"
            )

    def recall_critical_intent(
        self,
        memory_id: str,
        justification: str
    ) -> bytes:
        """
        Recall an intent from Memory Vault with full security gates.

        Args:
            memory_id: The Memory Vault memory ID
            justification: Human-readable justification for recall

        Returns:
            The intent content as bytes
        """
        vault = self._get_vault()
        return vault.recall_memory(memory_id, justification=justification)

    def classify_intent(
        self,
        intent_name: str,
        intent_reasoning: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Determine the appropriate classification level for an intent.

        Args:
            intent_name: Name of the intent
            intent_reasoning: Reasoning content
            metadata: Additional metadata

        Returns:
            Classification level (0-5)
        """
        # Simple rule-based classification
        # In production, this could use ML or more sophisticated logic

        name_lower = intent_name.lower()
        reasoning_lower = intent_reasoning.lower()

        # Critical keywords indicate high classification
        critical_keywords = ["seed", "key", "password", "credential", "recovery"]
        if any(keyword in name_lower or keyword in reasoning_lower
               for keyword in critical_keywords):
            return ClassificationLevel.RECOVERY_SEED

        # Strategic keywords
        strategic_keywords = ["goal", "principle", "strategy", "mission"]
        if any(keyword in name_lower or keyword in reasoning_lower
               for keyword in strategic_keywords):
            return ClassificationLevel.LONG_TERM_GOAL

        # Failure analysis
        if "failed" in name_lower or "lesson" in reasoning_lower:
            return ClassificationLevel.FAILED_PATH_LESSON

        # Learning
        if "learned" in name_lower or "heuristic" in reasoning_lower:
            return ClassificationLevel.LEARNED_HEURISTIC

        # Default to transient
        return ClassificationLevel.TRANSIENT

    def should_use_vault(self, classification: int) -> bool:
        """
        Determine if an intent should be stored in Memory Vault.

        Args:
            classification: The classification level

        Returns:
            True if should use vault, False for local storage
        """
        # Only use vault for classification >= 2
        return classification >= ClassificationLevel.FAILED_PATH_LESSON

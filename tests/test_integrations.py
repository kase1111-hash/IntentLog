"""Tests for IntentLog integrations"""
import pytest
from intentlog.integrations.memory_vault import (
    MemoryVaultIntegration,
    ClassificationLevel,
    MemoryVaultConfig
)


def test_memory_vault_config():
    """Test MemoryVaultConfig initialization"""
    config = MemoryVaultConfig()

    assert config.vault_path == ".memory_vault"
    assert config.encryption_profile == "default-passphrase"
    assert config.enable_tpm is False
    assert config.default_classification == 2


def test_memory_vault_integration_init():
    """Test MemoryVaultIntegration initialization"""
    integration = MemoryVaultIntegration()

    assert integration.config is not None
    assert integration._vault is None  # Lazy loaded


def test_classify_intent_critical():
    """Test classification of critical intents"""
    integration = MemoryVaultIntegration()

    # Test seed/key detection
    classification = integration.classify_intent(
        intent_name="store_recovery_seed",
        intent_reasoning="Storing recovery seed for system"
    )
    assert classification == ClassificationLevel.RECOVERY_SEED

    # Test password detection
    classification = integration.classify_intent(
        intent_name="save_password",
        intent_reasoning="User password storage"
    )
    assert classification == ClassificationLevel.RECOVERY_SEED


def test_classify_intent_strategic():
    """Test classification of strategic intents"""
    integration = MemoryVaultIntegration()

    classification = integration.classify_intent(
        intent_name="set_long_term_goal",
        intent_reasoning="Establishing strategic direction"
    )
    assert classification == ClassificationLevel.LONG_TERM_GOAL


def test_classify_intent_failure():
    """Test classification of failure lessons"""
    integration = MemoryVaultIntegration()

    classification = integration.classify_intent(
        intent_name="record_failed_approach",
        intent_reasoning="This approach failed due to X"
    )
    assert classification == ClassificationLevel.FAILED_PATH_LESSON


def test_classify_intent_learning():
    """Test classification of learned heuristics"""
    integration = MemoryVaultIntegration()

    classification = integration.classify_intent(
        intent_name="learned_pattern",
        intent_reasoning="Learned heuristic for optimization"
    )
    assert classification == ClassificationLevel.LEARNED_HEURISTIC


def test_classify_intent_transient():
    """Test classification of transient intents"""
    integration = MemoryVaultIntegration()

    classification = integration.classify_intent(
        intent_name="simple_action",
        intent_reasoning="Just a simple action"
    )
    assert classification == ClassificationLevel.TRANSIENT


def test_should_use_vault():
    """Test vault usage decision"""
    integration = MemoryVaultIntegration()

    # Should use vault for classification >= 2
    assert integration.should_use_vault(ClassificationLevel.RECOVERY_SEED) is True
    assert integration.should_use_vault(ClassificationLevel.LONG_TERM_GOAL) is True
    assert integration.should_use_vault(ClassificationLevel.FAILED_PATH_LESSON) is True

    # Should not use vault for low classification
    assert integration.should_use_vault(ClassificationLevel.LEARNED_HEURISTIC) is False
    assert integration.should_use_vault(ClassificationLevel.TRANSIENT) is False


def test_classification_levels():
    """Test ClassificationLevel enum values"""
    assert ClassificationLevel.TRANSIENT == 0
    assert ClassificationLevel.LEARNED_HEURISTIC == 1
    assert ClassificationLevel.FAILED_PATH_LESSON == 2
    assert ClassificationLevel.LONG_TERM_GOAL == 3
    assert ClassificationLevel.RECOVERY_SEED == 5

"""
Tests for LLM-based Intent Classification

Tests the semantic classification system that replaces keyword-based
classification with LLM-powered understanding.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from intentlog.integrations.llm_classifier import (
    LLMIntentClassifier,
    IntentCategory,
    ClassificationResult,
    classify_intent_with_llm,
)
from intentlog.core import Intent


# ============================================================================
# IntentCategory Tests
# ============================================================================

class TestIntentCategory:
    """Tests for IntentCategory enum"""

    def test_all_categories_exist(self):
        """All expected categories are defined"""
        expected = [
            "transient", "learned", "failure", "strategic", "critical",
            "architecture", "security", "compliance", "performance", "ux"
        ]
        for cat in expected:
            assert IntentCategory(cat) is not None

    def test_category_values(self):
        """Category values are correct"""
        assert IntentCategory.TRANSIENT.value == "transient"
        assert IntentCategory.RECOVERY_SEED.value == "critical"
        assert IntentCategory.LONG_TERM_GOAL.value == "strategic"
        assert IntentCategory.ARCHITECTURE.value == "architecture"


# ============================================================================
# ClassificationResult Tests
# ============================================================================

class TestClassificationResult:
    """Tests for ClassificationResult dataclass"""

    def test_create_result(self):
        """Create basic classification result"""
        result = ClassificationResult(
            category=IntentCategory.ARCHITECTURE,
            confidence=0.85,
            classification_level=2,
            reasoning="This is an architectural decision",
        )
        assert result.category == IntentCategory.ARCHITECTURE
        assert result.confidence == 0.85
        assert result.classification_level == 2
        assert result.reasoning == "This is an architectural decision"
        assert result.secondary_categories == []
        assert result.sensitivity_score == 0.0
        assert result.retention_priority == "normal"

    def test_result_with_all_fields(self):
        """Create result with all fields"""
        result = ClassificationResult(
            category=IntentCategory.SECURITY,
            confidence=0.9,
            classification_level=3,
            reasoning="Security-critical decision",
            secondary_categories=[IntentCategory.COMPLIANCE],
            sensitivity_score=0.8,
            retention_priority="high",
            suggested_tags=["security", "auth"],
            model="gpt-4",
            cached=True,
        )
        assert result.sensitivity_score == 0.8
        assert result.retention_priority == "high"
        assert len(result.suggested_tags) == 2
        assert result.model == "gpt-4"
        assert result.cached is True

    def test_result_to_dict(self):
        """Convert result to dictionary"""
        result = ClassificationResult(
            category=IntentCategory.LEARNED_HEURISTIC,
            confidence=0.75,
            classification_level=1,
            reasoning="Learned pattern",
            secondary_categories=[IntentCategory.PERFORMANCE],
            suggested_tags=["optimization"],
        )
        d = result.to_dict()

        assert d["category"] == "learned"
        assert d["confidence"] == 0.75
        assert d["classification_level"] == 1
        assert d["reasoning"] == "Learned pattern"
        assert d["secondary_categories"] == ["performance"]
        assert d["suggested_tags"] == ["optimization"]

    def test_result_from_dict(self):
        """Create result from dictionary"""
        data = {
            "category": "failure",
            "confidence": 0.8,
            "classification_level": 2,
            "reasoning": "Failure analysis",
            "secondary_categories": ["learned"],
            "sensitivity_score": 0.3,
            "retention_priority": "high",
            "suggested_tags": ["postmortem"],
            "model": "claude-3",
            "cached": False,
        }
        result = ClassificationResult.from_dict(data)

        assert result.category == IntentCategory.FAILED_PATH_LESSON
        assert result.confidence == 0.8
        assert result.classification_level == 2
        assert IntentCategory.LEARNED_HEURISTIC in result.secondary_categories


# ============================================================================
# LLMIntentClassifier Tests
# ============================================================================

class TestLLMIntentClassifier:
    """Tests for LLMIntentClassifier"""

    def _create_mock_provider(self, response_content: str) -> Mock:
        """Create a mock LLM provider with specified response"""
        provider = Mock()
        response = Mock()
        response.content = response_content
        response.model = "test-model"
        provider.complete.return_value = response
        return provider

    def test_classifier_creation(self):
        """Create classifier with provider"""
        provider = Mock()
        classifier = LLMIntentClassifier(provider)

        assert classifier.provider == provider
        assert classifier.cache_classifications is True
        assert classifier.fallback_to_keywords is True

    def test_classifier_options(self):
        """Classifier respects configuration options"""
        provider = Mock()
        classifier = LLMIntentClassifier(
            provider,
            cache_classifications=False,
            fallback_to_keywords=False,
        )

        assert classifier.cache_classifications is False
        assert classifier.fallback_to_keywords is False

    def test_classify_transient(self):
        """Classify transient intent"""
        response = """CATEGORY: transient
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.85
SENSITIVITY: 0.1
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: routine, temporary

REASONING:
This is a routine operation with no long-term significance."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify(
            "update-config",
            "Updating the configuration file for local testing"
        )

        assert result.category == IntentCategory.TRANSIENT
        assert result.classification_level == 0
        assert result.confidence == 0.85
        assert result.retention_priority == "low"
        assert "routine" in result.suggested_tags

    def test_classify_architecture(self):
        """Classify architectural decision"""
        response = """CATEGORY: architecture
CLASSIFICATION_LEVEL: 2
CONFIDENCE: 0.92
SENSITIVITY: 0.2
RETENTION: high
SECONDARY_CATEGORIES: performance
SUGGESTED_TAGS: architecture, microservices, scaling

REASONING:
This is a significant architectural decision about system design that should be preserved for future reference."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify(
            "adopt-microservices",
            "We're moving to microservices architecture for better scaling"
        )

        assert result.category == IntentCategory.ARCHITECTURE
        assert result.classification_level == 2
        assert result.confidence == 0.92
        assert IntentCategory.PERFORMANCE in result.secondary_categories
        assert "microservices" in result.suggested_tags

    def test_classify_security_critical(self):
        """Classify security-critical intent"""
        response = """CATEGORY: critical
CLASSIFICATION_LEVEL: 5
CONFIDENCE: 0.98
SENSITIVITY: 1.0
RETENTION: critical
SECONDARY_CATEGORIES: security
SUGGESTED_TAGS: security, credentials, critical

REASONING:
This intent contains critical security information that must be protected."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify(
            "store-api-key",
            "Storing the API key for production access"
        )

        assert result.category == IntentCategory.RECOVERY_SEED
        assert result.classification_level == 5
        assert result.sensitivity_score == 1.0
        assert result.retention_priority == "critical"

    def test_classify_with_cache(self):
        """Classification results are cached"""
        response = """CATEGORY: learned
CLASSIFICATION_LEVEL: 1
CONFIDENCE: 0.8
SENSITIVITY: 0.0
RETENTION: normal
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: learning

REASONING:
This is a learned heuristic."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider, cache_classifications=True)

        # First call
        result1 = classifier.classify("test", "Test reasoning")
        assert result1.cached is False
        assert provider.complete.call_count == 1

        # Second call (should be cached)
        result2 = classifier.classify("test", "Test reasoning")
        assert result2.cached is True
        assert provider.complete.call_count == 1  # No additional call

    def test_classify_without_cache(self):
        """Classification without caching"""
        response = """CATEGORY: transient
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.7
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Basic intent."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider, cache_classifications=False)

        # Multiple calls should all hit the provider
        classifier.classify("test", "Test reasoning")
        classifier.classify("test", "Test reasoning")
        assert provider.complete.call_count == 2

    def test_classify_intent_object(self):
        """Classify Intent object directly"""
        response = """CATEGORY: strategic
CLASSIFICATION_LEVEL: 3
CONFIDENCE: 0.9
SENSITIVITY: 0.3
RETENTION: high
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: roadmap

REASONING:
Strategic goal."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        intent = Intent(
            intent_name="Q1-roadmap",
            intent_reasoning="Planning Q1 product roadmap and priorities",
        )

        result = classifier.classify_intent(intent)
        assert result.category == IntentCategory.LONG_TERM_GOAL
        assert result.classification_level == 3

    def test_batch_classify(self):
        """Batch classify multiple intents"""
        response = """CATEGORY: transient
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.7
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Basic."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        intents = [
            Intent(intent_name="test1", intent_reasoning="Test 1"),
            Intent(intent_name="test2", intent_reasoning="Test 2"),
            Intent(intent_name="test3", intent_reasoning="Test 3"),
        ]

        results = classifier.batch_classify(intents)
        assert len(results) == 3

    def test_cache_stats(self):
        """Get cache statistics"""
        provider = Mock()
        classifier = LLMIntentClassifier(provider, cache_classifications=True)

        stats = classifier.get_cache_stats()
        assert stats["size"] == 0
        assert stats["enabled"] is True

    def test_clear_cache(self):
        """Clear classification cache"""
        response = """CATEGORY: transient
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.7
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Basic."""

        provider = self._create_mock_provider(response)
        classifier = LLMIntentClassifier(provider)

        classifier.classify("test", "reasoning")
        assert classifier.get_cache_stats()["size"] == 1

        classifier.clear_cache()
        assert classifier.get_cache_stats()["size"] == 0


# ============================================================================
# Keyword Fallback Tests
# ============================================================================

class TestKeywordFallback:
    """Tests for keyword-based fallback classification"""

    def test_fallback_on_error(self):
        """Falls back to keywords on LLM error"""
        provider = Mock()
        provider.complete.side_effect = Exception("LLM error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "encrypt-data",
            "Need to encrypt the user data"
        )

        assert result.category == IntentCategory.SECURITY
        assert "Fallback" in result.reasoning

    def test_fallback_critical_keywords(self):
        """Fallback detects critical keywords"""
        provider = Mock()
        provider.complete.side_effect = Exception("error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "store-recovery-seed",
            "Storing the recovery seed phrase"
        )

        assert result.category == IntentCategory.RECOVERY_SEED
        assert result.classification_level == 5
        assert result.sensitivity_score == 1.0

    def test_fallback_strategic_keywords(self):
        """Fallback detects strategic keywords"""
        provider = Mock()
        provider.complete.side_effect = Exception("error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "define-mission",
            "Defining the company mission statement"
        )

        assert result.category == IntentCategory.LONG_TERM_GOAL
        assert result.classification_level == 3

    def test_fallback_failure_keywords(self):
        """Fallback detects failure/lesson keywords"""
        provider = Mock()
        provider.complete.side_effect = Exception("error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "post-mortem",
            "This approach failed due to scaling issues"
        )

        assert result.category == IntentCategory.FAILED_PATH_LESSON
        assert result.classification_level == 2

    def test_fallback_architecture_keywords(self):
        """Fallback detects architecture keywords"""
        provider = Mock()
        provider.complete.side_effect = Exception("error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "system-design",
            "Designing the new architecture for the backend"
        )

        assert result.category == IntentCategory.ARCHITECTURE
        assert result.classification_level == 2

    def test_fallback_default_transient(self):
        """Fallback defaults to transient for unknown"""
        provider = Mock()
        provider.complete.side_effect = Exception("error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=True)

        result = classifier.classify(
            "update",
            "Updating some values"
        )

        assert result.category == IntentCategory.TRANSIENT
        assert result.classification_level == 0

    def test_no_fallback_raises(self):
        """Without fallback, errors are raised"""
        provider = Mock()
        provider.complete.side_effect = Exception("LLM error")

        classifier = LLMIntentClassifier(provider, fallback_to_keywords=False)

        with pytest.raises(Exception, match="LLM error"):
            classifier.classify("test", "test")


# ============================================================================
# Memory Vault Integration Tests
# ============================================================================

class TestMemoryVaultIntegration:
    """Tests for Memory Vault compatibility"""

    def _create_mock_provider(self, level: int) -> Mock:
        """Create mock provider returning specific level"""
        response = f"""CATEGORY: learned
CLASSIFICATION_LEVEL: {level}
CONFIDENCE: 0.8
SENSITIVITY: 0.0
RETENTION: normal
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Test classification."""

        provider = Mock()
        resp = Mock()
        resp.content = response
        resp.model = "test"
        provider.complete.return_value = resp
        return provider

    def test_get_memory_vault_level(self):
        """Get Memory Vault compatible level"""
        provider = self._create_mock_provider(3)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify("test", "test")
        level = classifier.get_memory_vault_level(result)

        assert level == 3

    def test_should_persist_level_0(self):
        """Level 0 should not persist"""
        provider = self._create_mock_provider(0)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify("test", "test")
        assert classifier.should_persist(result) is False

    def test_should_persist_level_1(self):
        """Level 1+ should persist"""
        provider = self._create_mock_provider(1)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify("test", "test")
        assert classifier.should_persist(result) is True

    def test_should_use_vault_level_1(self):
        """Level 1 should not use vault"""
        provider = self._create_mock_provider(1)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify("test", "test")
        assert classifier.should_use_vault(result) is False

    def test_should_use_vault_level_2(self):
        """Level 2+ should use vault"""
        provider = self._create_mock_provider(2)
        classifier = LLMIntentClassifier(provider)

        result = classifier.classify("test", "test")
        assert classifier.should_use_vault(result) is True


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunction:
    """Tests for classify_intent_with_llm function"""

    def test_convenience_function(self):
        """classify_intent_with_llm works correctly"""
        response = """CATEGORY: compliance
CLASSIFICATION_LEVEL: 2
CONFIDENCE: 0.85
SENSITIVITY: 0.5
RETENTION: high
SECONDARY_CATEGORIES: security
SUGGESTED_TAGS: gdpr, privacy

REASONING:
GDPR compliance decision."""

        provider = Mock()
        resp = Mock()
        resp.content = response
        resp.model = "test"
        provider.complete.return_value = resp

        result = classify_intent_with_llm(
            provider,
            "gdpr-compliance",
            "Implementing GDPR data deletion requirements"
        )

        assert result.category == IntentCategory.COMPLIANCE
        assert result.classification_level == 2
        assert "gdpr" in result.suggested_tags


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing"""

    def _create_classifier_with_response(self, response: str) -> LLMIntentClassifier:
        """Create classifier with mock response"""
        provider = Mock()
        resp = Mock()
        resp.content = response
        resp.model = "test"
        provider.complete.return_value = resp
        return LLMIntentClassifier(provider)

    def test_parse_minimal_response(self):
        """Parse minimal valid response"""
        response = """CATEGORY: transient
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.5
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Basic intent."""

        classifier = self._create_classifier_with_response(response)
        result = classifier.classify("test", "test")

        assert result.category == IntentCategory.TRANSIENT
        assert result.classification_level == 0

    def test_parse_multiline_reasoning(self):
        """Parse multi-line reasoning"""
        response = """CATEGORY: architecture
CLASSIFICATION_LEVEL: 2
CONFIDENCE: 0.9
SENSITIVITY: 0.2
RETENTION: high
SECONDARY_CATEGORIES: performance, security
SUGGESTED_TAGS: design, scaling

REASONING:
This is a complex architectural decision.
It affects multiple system components.
Should be preserved for future reference."""

        classifier = self._create_classifier_with_response(response)
        result = classifier.classify("test", "test")

        assert "complex architectural" in result.reasoning
        assert "future reference" in result.reasoning

    def test_parse_invalid_category_defaults(self):
        """Invalid category defaults to transient"""
        response = """CATEGORY: invalid_category
CLASSIFICATION_LEVEL: 0
CONFIDENCE: 0.5
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Test."""

        classifier = self._create_classifier_with_response(response)
        result = classifier.classify("test", "test")

        # Should default to transient (from fallback)
        assert result.category == IntentCategory.TRANSIENT

    def test_parse_invalid_level_defaults(self):
        """Invalid level is ignored"""
        response = """CATEGORY: learned
CLASSIFICATION_LEVEL: 99
CONFIDENCE: 0.5
SENSITIVITY: 0.0
RETENTION: low
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Test."""

        classifier = self._create_classifier_with_response(response)
        result = classifier.classify("test", "test")

        # Should default to 0 (invalid level ignored)
        assert result.classification_level == 0

    def test_parse_confidence_bounds(self):
        """Confidence is bounded to 0-1"""
        response = """CATEGORY: learned
CLASSIFICATION_LEVEL: 1
CONFIDENCE: 1.5
SENSITIVITY: 0.0
RETENTION: normal
SECONDARY_CATEGORIES: none
SUGGESTED_TAGS: none

REASONING:
Test."""

        classifier = self._create_classifier_with_response(response)
        result = classifier.classify("test", "test")

        # Should be clamped to 1.0
        assert result.confidence == 1.0

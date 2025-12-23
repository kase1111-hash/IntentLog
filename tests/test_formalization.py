"""
Tests for Deferred Formalization Feature

Tests the ability to derive formal code, rules, and heuristics from prose intent.
"""

import pytest
from datetime import datetime
from pathlib import Path

from intentlog.core import Intent
from intentlog.semantic import (
    FormalizationType,
    FormalizedOutput,
    ProvenanceRecord,
    SemanticEngine,
)
from intentlog.llm.provider import MockProvider


class TestFormalizationType:
    """Tests for FormalizationType enum"""

    def test_formalization_types_exist(self):
        """Test that all formalization types exist"""
        assert FormalizationType.CODE.value == "code"
        assert FormalizationType.RULES.value == "rules"
        assert FormalizationType.HEURISTICS.value == "heuristics"
        assert FormalizationType.SCHEMA.value == "schema"
        assert FormalizationType.CONFIG.value == "config"
        assert FormalizationType.SPEC.value == "spec"
        assert FormalizationType.TESTS.value == "tests"

    def test_formalization_type_from_string(self):
        """Test creating FormalizationType from string"""
        assert FormalizationType("code") == FormalizationType.CODE
        assert FormalizationType("rules") == FormalizationType.RULES


class TestProvenanceRecord:
    """Tests for ProvenanceRecord"""

    def test_provenance_creation(self):
        """Test creating a provenance record"""
        provenance = ProvenanceRecord(
            source_intent_ids=["id1", "id2"],
            source_reasoning="Test reasoning",
            formalized_at=datetime.now(),
            model="test-model",
            formalization_type=FormalizationType.CODE,
        )
        assert len(provenance.source_intent_ids) == 2
        assert provenance.source_reasoning == "Test reasoning"
        assert provenance.model == "test-model"
        assert provenance.formalization_type == FormalizationType.CODE

    def test_provenance_to_dict(self):
        """Test converting provenance to dict"""
        now = datetime.now()
        provenance = ProvenanceRecord(
            source_intent_ids=["id1"],
            source_reasoning="Test",
            formalized_at=now,
            model="test-model",
            formalization_type=FormalizationType.RULES,
            parameters={"language": "python"},
        )
        d = provenance.to_dict()
        assert d["source_intent_ids"] == ["id1"]
        assert d["source_reasoning"] == "Test"
        assert d["model"] == "test-model"
        assert d["formalization_type"] == "rules"
        assert d["parameters"]["language"] == "python"

    def test_provenance_from_dict(self):
        """Test creating provenance from dict"""
        data = {
            "source_intent_ids": ["id1", "id2"],
            "source_reasoning": "Test reasoning",
            "formalized_at": "2025-01-01T12:00:00",
            "model": "test-model",
            "formalization_type": "code",
            "parameters": {},
        }
        provenance = ProvenanceRecord.from_dict(data)
        assert provenance.source_intent_ids == ["id1", "id2"]
        assert provenance.model == "test-model"
        assert provenance.formalization_type == FormalizationType.CODE


class TestFormalizedOutput:
    """Tests for FormalizedOutput"""

    def test_formalized_output_creation(self):
        """Test creating formalized output"""
        provenance = ProvenanceRecord(
            source_intent_ids=["id1"],
            source_reasoning="Test",
            formalized_at=datetime.now(),
            model="test-model",
            formalization_type=FormalizationType.CODE,
        )
        output = FormalizedOutput(
            content="def hello(): pass",
            formalization_type=FormalizationType.CODE,
            language="python",
            explanation="A simple function",
            provenance=provenance,
            confidence=0.9,
            warnings=["May need error handling"],
        )
        assert output.content == "def hello(): pass"
        assert output.language == "python"
        assert output.confidence == 0.9
        assert len(output.warnings) == 1

    def test_formalized_output_to_dict(self):
        """Test converting formalized output to dict"""
        provenance = ProvenanceRecord(
            source_intent_ids=["id1"],
            source_reasoning="Test",
            formalized_at=datetime.now(),
            model="test-model",
            formalization_type=FormalizationType.CODE,
        )
        output = FormalizedOutput(
            content="code here",
            formalization_type=FormalizationType.CODE,
            language="python",
            explanation="explanation",
            provenance=provenance,
        )
        d = output.to_dict()
        assert d["content"] == "code here"
        assert d["formalization_type"] == "code"
        assert d["language"] == "python"
        assert "provenance" in d

    def test_formalized_output_from_dict(self):
        """Test creating formalized output from dict"""
        data = {
            "content": "test code",
            "formalization_type": "rules",
            "language": None,
            "explanation": "test explanation",
            "provenance": {
                "source_intent_ids": ["id1"],
                "source_reasoning": "reasoning",
                "formalized_at": "2025-01-01T12:00:00",
                "model": "test-model",
                "formalization_type": "rules",
                "parameters": {},
            },
            "confidence": 0.85,
            "warnings": [],
        }
        output = FormalizedOutput.from_dict(data)
        assert output.content == "test code"
        assert output.formalization_type == FormalizationType.RULES
        assert output.confidence == 0.85


class TestSemanticEngineFormalization:
    """Tests for SemanticEngine formalization methods"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider"""
        return MockProvider()

    @pytest.fixture
    def engine(self, mock_provider):
        """Create a semantic engine with mock provider"""
        return SemanticEngine(mock_provider)

    @pytest.fixture
    def sample_intent(self):
        """Create a sample intent for testing"""
        return Intent(
            intent_id="test-intent-001",
            intent_name="Authentication Design",
            intent_reasoning="We need to implement JWT-based authentication with refresh tokens. "
                           "The system should support both mobile and web clients, with token "
                           "expiration of 15 minutes and refresh token validity of 7 days.",
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def intent_chain(self):
        """Create a chain of intents for testing"""
        return [
            Intent(
                intent_id="intent-001",
                intent_name="Initial Auth Design",
                intent_reasoning="Start with basic JWT authentication.",
                timestamp=datetime(2025, 1, 1, 10, 0),
            ),
            Intent(
                intent_id="intent-002",
                intent_name="Add Refresh Tokens",
                intent_reasoning="Add refresh token support for better UX.",
                timestamp=datetime(2025, 1, 1, 11, 0),
            ),
            Intent(
                intent_id="intent-003",
                intent_name="Security Hardening",
                intent_reasoning="Add rate limiting and token revocation.",
                timestamp=datetime(2025, 1, 1, 12, 0),
            ),
        ]

    def test_formalize_returns_output(self, engine, sample_intent, mock_provider):
        """Test that formalize returns FormalizedOutput"""
        # Set up mock response
        mock_provider.set_responses([
            """CODE:
```python
def authenticate(token):
    # JWT authentication
    return verify_token(token)
```

EXPLANATION:
This implements JWT-based authentication.

CONFIDENCE: 0.9

WARNINGS:
None"""
        ])

        result = engine.formalize(sample_intent)

        assert isinstance(result, FormalizedOutput)
        assert result.formalization_type == FormalizationType.CODE
        assert result.language == "python"
        assert result.provenance is not None
        assert sample_intent.intent_id in result.provenance.source_intent_ids

    def test_formalize_with_rules_type(self, engine, sample_intent, mock_provider):
        """Test formalization to rules"""
        mock_provider.set_responses([
            """RULES:
1. IF user provides credentials THEN generate JWT token
2. IF token expires THEN check refresh token validity
3. IF refresh token valid THEN issue new access token

EXPLANATION:
These rules cover the authentication flow.

CONFIDENCE: 0.85

WARNINGS:
- Does not cover revocation scenarios"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.RULES,
        )

        assert result.formalization_type == FormalizationType.RULES
        assert "IF" in result.content or len(result.content) > 0

    def test_formalize_with_heuristics_type(self, engine, sample_intent, mock_provider):
        """Test formalization to heuristics"""
        mock_provider.set_responses([
            """HEURISTICS:
1. Always validate JWT signature before trusting claims
2. Use short-lived access tokens (15 min)
3. Store refresh tokens securely

EXPLANATION:
Security best practices for JWT.

CONFIDENCE: 0.8

WARNINGS:
Environment-specific considerations may apply"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.HEURISTICS,
        )

        assert result.formalization_type == FormalizationType.HEURISTICS

    def test_formalize_with_custom_language(self, engine, sample_intent, mock_provider):
        """Test formalization with custom language"""
        mock_provider.set_responses([
            """CODE:
```typescript
function authenticate(token: string): boolean {
    return verifyToken(token);
}
```

EXPLANATION:
TypeScript implementation.

CONFIDENCE: 0.88

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.CODE,
            language="typescript",
        )

        assert result.language == "typescript"

    def test_formalize_with_context(self, engine, sample_intent, mock_provider):
        """Test formalization with additional context"""
        mock_provider.set_responses([
            """CODE:
```python
# With context
def auth_with_context():
    pass
```

EXPLANATION:
Considers additional context.

CONFIDENCE: 0.75

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            context="The system uses PostgreSQL for storage.",
        )

        assert result.provenance.parameters.get("has_additional_context") is False  # No additional_context intents

    def test_formalize_chain_returns_output(self, engine, intent_chain, mock_provider):
        """Test that formalize_chain returns FormalizedOutput"""
        mock_provider.set_responses([
            """FORMALIZED OUTPUT:
```python
class AuthenticationService:
    def __init__(self):
        self.rate_limiter = RateLimiter()

    def authenticate(self, token):
        self.rate_limiter.check()
        return self.verify_token(token)
```

PROVENANCE SUMMARY:
The chain evolved from basic JWT to include refresh tokens and security hardening.

CONFIDENCE: 0.82

WARNINGS:
Implementation is a simplified version"""
        ])

        result = engine.formalize_chain(intent_chain)

        assert isinstance(result, FormalizedOutput)
        assert len(result.provenance.source_intent_ids) == 3
        assert "→" in result.provenance.source_reasoning

    def test_formalize_chain_empty_raises_error(self, engine):
        """Test that formalize_chain raises error with empty list"""
        with pytest.raises(ValueError, match="At least one intent is required"):
            engine.formalize_chain([])

    def test_formalize_chain_with_rules(self, engine, intent_chain, mock_provider):
        """Test formalize_chain with rules output"""
        mock_provider.set_responses([
            """FORMALIZED OUTPUT:
1. WHEN authenticating THEN verify JWT signature
2. WHEN token expires THEN check refresh token
3. WHEN rate limit exceeded THEN reject request

PROVENANCE SUMMARY:
Rules derived from the evolution of auth requirements.

CONFIDENCE: 0.78

WARNINGS:
None"""
        ])

        result = engine.formalize_chain(
            intent_chain,
            formalization_type=FormalizationType.RULES,
        )

        assert result.formalization_type == FormalizationType.RULES
        assert result.provenance.parameters.get("chain_length") == 3

    def test_formalize_schema_type(self, engine, sample_intent, mock_provider):
        """Test formalization to schema"""
        mock_provider.set_responses([
            """SCHEMA:
```json_schema
{
    "type": "object",
    "properties": {
        "token": {"type": "string"},
        "expires_at": {"type": "string", "format": "date-time"}
    }
}
```

EXPLANATION:
JSON Schema for JWT token.

CONFIDENCE: 0.9

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.SCHEMA,
        )

        assert result.formalization_type == FormalizationType.SCHEMA
        assert result.language == "json_schema"

    def test_formalize_config_type(self, engine, sample_intent, mock_provider):
        """Test formalization to config"""
        mock_provider.set_responses([
            """CONFIG:
```yaml
auth:
  jwt:
    expiration: 900  # 15 minutes
    algorithm: RS256
  refresh:
    validity: 604800  # 7 days
```

EXPLANATION:
YAML config for auth settings.

CONFIDENCE: 0.95

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.CONFIG,
        )

        assert result.formalization_type == FormalizationType.CONFIG
        assert result.language == "yaml"

    def test_formalize_spec_type(self, engine, sample_intent, mock_provider):
        """Test formalization to specification"""
        mock_provider.set_responses([
            """SPECIFICATION:
1. The system SHALL implement JWT-based authentication.
2. Access tokens SHALL expire after 15 minutes.
3. Refresh tokens SHALL be valid for 7 days.

EXPLANATION:
Formal requirements specification.

CONFIDENCE: 0.88

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.SPEC,
        )

        assert result.formalization_type == FormalizationType.SPEC

    def test_formalize_tests_type(self, engine, sample_intent, mock_provider):
        """Test formalization to tests"""
        mock_provider.set_responses([
            """TESTS:
```python
def test_jwt_authentication():
    token = create_token(user_id=1)
    assert verify_token(token) is True

def test_token_expiration():
    expired = create_token(user_id=1, expired=True)
    assert verify_token(expired) is False
```

EXPLANATION:
Test cases for JWT auth.

CONFIDENCE: 0.85

WARNINGS:
May need additional edge case tests"""
        ])

        result = engine.formalize(
            sample_intent,
            formalization_type=FormalizationType.TESTS,
        )

        assert result.formalization_type == FormalizationType.TESTS
        assert "test_" in result.content or len(result.content) > 0

    def test_formalize_from_search(self, engine, intent_chain, mock_provider):
        """Test formalize_from_search finds and formalizes intents"""
        # Set up embeddings for search
        mock_provider.set_responses([
            """FORMALIZED OUTPUT:
```python
def authenticate():
    pass
```

PROVENANCE SUMMARY:
Derived from search results.

CONFIDENCE: 0.8

WARNINGS:
None"""
        ])

        result = engine.formalize_from_search(
            "authentication security",
            intent_chain,
            formalization_type=FormalizationType.CODE,
        )

        assert isinstance(result, FormalizedOutput)

    def test_formalize_provenance_tracking(self, engine, sample_intent, mock_provider):
        """Test that provenance is correctly tracked"""
        mock_provider.set_responses([
            """CODE:
```python
pass
```

EXPLANATION:
Test

CONFIDENCE: 0.7

WARNINGS:
None"""
        ])

        result = engine.formalize(sample_intent)

        provenance = result.provenance
        assert sample_intent.intent_id in provenance.source_intent_ids
        assert provenance.formalized_at is not None
        assert provenance.model == "mock-model"
        assert provenance.formalization_type == FormalizationType.CODE

    def test_formalize_with_additional_intents(self, engine, sample_intent, intent_chain, mock_provider):
        """Test formalization with additional context intents"""
        mock_provider.set_responses([
            """CODE:
```python
# With related context
def auth():
    pass
```

EXPLANATION:
Uses related intents for context.

CONFIDENCE: 0.85

WARNINGS:
None"""
        ])

        result = engine.formalize(
            sample_intent,
            additional_context=intent_chain,
        )

        # Should include all intent IDs in provenance
        assert len(result.provenance.source_intent_ids) == 4  # sample + 3 chain
        assert result.provenance.parameters.get("has_additional_context") is True


class TestFormalizationParsing:
    """Tests for response parsing in formalization"""

    @pytest.fixture
    def engine(self):
        """Create engine with mock provider"""
        return SemanticEngine(MockProvider())

    def test_parse_confidence_decimal(self, engine):
        """Test parsing decimal confidence"""
        content = """CODE:
test

EXPLANATION:
test

CONFIDENCE: 0.85

WARNINGS:
None"""
        _, _, confidence, _ = engine._parse_formalized_response(
            content, FormalizationType.CODE
        )
        assert confidence == 0.85

    def test_parse_confidence_percentage(self, engine):
        """Test parsing percentage confidence"""
        content = """CODE:
test

EXPLANATION:
test

CONFIDENCE: 85%

WARNINGS:
None"""
        _, _, confidence, _ = engine._parse_formalized_response(
            content, FormalizationType.CODE
        )
        assert confidence == 0.85

    def test_parse_warnings(self, engine):
        """Test parsing warnings"""
        content = """CODE:
test

EXPLANATION:
test

CONFIDENCE: 0.8

WARNINGS:
- First warning
- Second warning"""
        _, _, _, warnings = engine._parse_formalized_response(
            content, FormalizationType.CODE
        )
        assert len(warnings) == 2
        assert "First warning" in warnings[0]

    def test_parse_code_block(self, engine):
        """Test parsing code blocks"""
        content = """CODE:
```python
def hello():
    return "world"
```

EXPLANATION:
Simple function.

CONFIDENCE: 0.9

WARNINGS:
None"""
        formalized, _, _, _ = engine._parse_formalized_response(
            content, FormalizationType.CODE
        )
        assert "def hello():" in formalized
        assert "return" in formalized


class TestFormalizationIntegration:
    """Integration tests for formalization"""

    def test_round_trip_serialization(self):
        """Test that FormalizedOutput can be serialized and deserialized"""
        provenance = ProvenanceRecord(
            source_intent_ids=["id1", "id2"],
            source_reasoning="Test reasoning",
            formalized_at=datetime.now(),
            model="test-model",
            formalization_type=FormalizationType.CODE,
            parameters={"language": "python"},
        )
        original = FormalizedOutput(
            content="def test(): pass",
            formalization_type=FormalizationType.CODE,
            language="python",
            explanation="A test function",
            provenance=provenance,
            confidence=0.9,
            warnings=["Warning 1", "Warning 2"],
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = FormalizedOutput.from_dict(data)

        # Verify
        assert restored.content == original.content
        assert restored.formalization_type == original.formalization_type
        assert restored.language == original.language
        assert restored.explanation == original.explanation
        assert restored.confidence == original.confidence
        assert restored.warnings == original.warnings
        assert len(restored.provenance.source_intent_ids) == 2

    def test_formalization_maintains_intent_relationship(self):
        """Test that formalization maintains relationship to source intents"""
        provider = MockProvider()
        provider.set_responses([
            """CODE:
```python
pass
```

EXPLANATION:
test

CONFIDENCE: 0.8

WARNINGS:
None"""
        ])
        engine = SemanticEngine(provider)

        intent = Intent(
            intent_id="unique-id-123",
            intent_name="Test Intent",
            intent_reasoning="Test reasoning for formalization",
            timestamp=datetime.now(),
        )

        result = engine.formalize(intent)

        # The output should reference the source intent
        assert intent.intent_id in result.provenance.source_intent_ids
        assert intent.intent_reasoning == result.provenance.source_reasoning

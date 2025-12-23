"""
Tests for LLM module
"""

import pytest
from datetime import datetime

from intentlog.llm.provider import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    EmbeddingResponse,
    LLMError,
    MockProvider,
)
from intentlog.llm.registry import (
    register_provider,
    get_provider,
    list_providers,
    _providers,
)
from intentlog.core import Intent
from intentlog.semantic import SemanticEngine, SemanticDiff


class TestLLMConfig:
    """Tests for LLMConfig"""

    def test_config_creation(self):
        """Test creating a config"""
        config = LLMConfig(provider="openai", model="gpt-4")
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7

    def test_config_get_api_key_direct(self):
        """Test getting API key from config"""
        config = LLMConfig(provider="openai", api_key="sk-test123")
        assert config.get_api_key() == "sk-test123"

    def test_config_get_api_key_env(self, monkeypatch):
        """Test getting API key from environment"""
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        config = LLMConfig(provider="openai", api_key_env="TEST_API_KEY")
        assert config.get_api_key() == "sk-from-env"

    def test_config_defaults(self):
        """Test config defaults"""
        config = LLMConfig(provider="test")
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.timeout == 30.0


class TestMockProvider:
    """Tests for MockProvider"""

    def test_mock_provider_creation(self):
        """Test creating mock provider"""
        provider = MockProvider()
        assert provider.name == "mock"
        assert provider.is_available()

    def test_mock_provider_complete(self):
        """Test mock completion"""
        provider = MockProvider()
        response = provider.complete("Hello, world!")

        assert isinstance(response, LLMResponse)
        assert response.model == "mock-model"
        assert len(response.content) > 0

    def test_mock_provider_set_responses(self):
        """Test setting mock responses"""
        provider = MockProvider()
        provider.set_responses(["Response 1", "Response 2"])

        response1 = provider.complete("First prompt")
        response2 = provider.complete("Second prompt")

        assert response1.content == "Response 1"
        assert response2.content == "Response 2"

    def test_mock_provider_embed(self):
        """Test mock embedding"""
        provider = MockProvider()
        response = provider.embed("Test text")

        assert isinstance(response, EmbeddingResponse)
        assert len(response.embedding) > 0
        assert all(0 <= x <= 1 for x in response.embedding)

    def test_mock_provider_deterministic_embedding(self):
        """Test that same text produces same embedding"""
        provider = MockProvider()

        embed1 = provider.embed("Test text")
        embed2 = provider.embed("Test text")

        assert embed1.embedding == embed2.embedding

    def test_mock_provider_different_embeddings(self):
        """Test that different text produces different embeddings"""
        provider = MockProvider()

        embed1 = provider.embed("Text one")
        embed2 = provider.embed("Text two")

        assert embed1.embedding != embed2.embedding


class TestProviderRegistry:
    """Tests for provider registry"""

    def test_register_provider(self):
        """Test registering a provider"""
        # Clear and re-register
        _providers.clear()
        register_provider("test", MockProvider)

        assert "test" in _providers
        assert _providers["test"] == MockProvider

    def test_get_provider(self):
        """Test getting a provider"""
        _providers.clear()
        register_provider("mock", MockProvider)

        config = LLMConfig(provider="mock")
        provider = get_provider(config)

        assert isinstance(provider, MockProvider)

    def test_get_unknown_provider(self):
        """Test getting unknown provider raises error"""
        _providers.clear()

        config = LLMConfig(provider="nonexistent")
        with pytest.raises(LLMError):
            get_provider(config)

    def test_list_providers(self):
        """Test listing providers"""
        _providers.clear()
        register_provider("mock", MockProvider)

        providers = list_providers()
        assert "mock" in providers


class TestSemanticEngine:
    """Tests for SemanticEngine"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider"""
        return MockProvider()

    @pytest.fixture
    def engine(self, mock_provider):
        """Create semantic engine with mock provider"""
        return SemanticEngine(mock_provider)

    @pytest.fixture
    def sample_intents(self):
        """Create sample intents for testing"""
        return [
            Intent(
                intent_id="1",
                intent_name="Architecture Decision",
                intent_reasoning="We chose microservices for better scalability",
                timestamp=datetime(2025, 1, 1, 10, 0),
            ),
            Intent(
                intent_id="2",
                intent_name="Database Choice",
                intent_reasoning="PostgreSQL selected for ACID compliance",
                timestamp=datetime(2025, 1, 2, 10, 0),
            ),
            Intent(
                intent_id="3",
                intent_name="API Design",
                intent_reasoning="REST API with OpenAPI documentation",
                timestamp=datetime(2025, 1, 3, 10, 0),
            ),
        ]

    def test_semantic_diff(self, engine):
        """Test semantic diff between two intents"""
        intent_a = Intent(
            intent_name="Initial Design",
            intent_reasoning="Using monolithic architecture for simplicity",
        )
        intent_b = Intent(
            intent_name="Updated Design",
            intent_reasoning="Switching to microservices for scalability",
        )

        # Set up mock response
        engine.provider.set_responses([
            "The design evolved from monolithic to microservices.\n"
            "- Changed from monolithic to microservices\n"
            "- Focus shifted to scalability"
        ])

        diff = engine.semantic_diff(intent_a, intent_b)

        assert isinstance(diff, SemanticDiff)
        assert diff.intent_a == intent_a
        assert diff.intent_b == intent_b
        assert len(diff.summary) > 0

    def test_semantic_search(self, engine, sample_intents):
        """Test semantic search"""
        results = engine.semantic_search(
            "database",
            sample_intents,
            top_k=2,
        )

        assert len(results) <= 2
        for result in results:
            assert result.intent in sample_intents
            assert 0 <= result.score <= 1

    def test_semantic_search_empty(self, engine):
        """Test semantic search with no intents"""
        results = engine.semantic_search("query", [])
        assert results == []

    def test_semantic_search_ranking(self, engine, sample_intents):
        """Test that results are ranked by score"""
        results = engine.semantic_search(
            "scalability microservices",
            sample_intents,
            top_k=3,
        )

        # Check scores are descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_similarity(self, engine):
        """Test cosine similarity calculation"""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert engine._cosine_similarity(a, b) == 1.0

        c = [1.0, 0.0, 0.0]
        d = [0.0, 1.0, 0.0]
        assert engine._cosine_similarity(c, d) == 0.0

    def test_embedding_caching(self, engine):
        """Test that embeddings are cached"""
        text = "Test text for caching"

        # First call
        embed1 = engine._get_embedding(text)

        # Second call should use cache
        embed2 = engine._get_embedding(text)

        assert embed1 == embed2

    def test_resolve_merge(self, engine):
        """Test merge resolution"""
        intent_a = Intent(
            intent_name="Use SQL",
            intent_reasoning="SQL databases are reliable",
        )
        intent_b = Intent(
            intent_name="Use NoSQL",
            intent_reasoning="NoSQL for flexibility",
        )

        engine.provider.set_responses([
            "MERGED REASONING:\n"
            "Use a hybrid approach with both SQL and NoSQL.\n\n"
            "EXPLANATION:\n"
            "Combining the reliability of SQL with NoSQL flexibility."
        ])

        resolution = engine.resolve_merge(intent_a, intent_b)

        assert len(resolution.resolved_reasoning) > 0
        assert resolution.intent_a == intent_a
        assert resolution.intent_b == intent_b


class TestLLMSettings:
    """Tests for LLMSettings in storage"""

    def test_llm_settings_creation(self):
        """Test creating LLM settings"""
        from intentlog.storage import LLMSettings

        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
        )
        assert settings.provider == "openai"
        assert settings.model == "gpt-4"

    def test_llm_settings_to_dict(self):
        """Test serializing LLM settings"""
        from intentlog.storage import LLMSettings

        settings = LLMSettings(
            provider="anthropic",
            model="claude-3",
            api_key_env="ANTHROPIC_KEY",
        )
        data = settings.to_dict()

        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-3"
        assert data["api_key_env"] == "ANTHROPIC_KEY"

    def test_llm_settings_from_dict(self):
        """Test deserializing LLM settings"""
        from intentlog.storage import LLMSettings

        data = {
            "provider": "ollama",
            "model": "llama2",
            "base_url": "http://localhost:11434",
        }
        settings = LLMSettings.from_dict(data)

        assert settings.provider == "ollama"
        assert settings.model == "llama2"
        assert settings.base_url == "http://localhost:11434"

    def test_llm_settings_is_configured(self):
        """Test is_configured check"""
        from intentlog.storage import LLMSettings

        empty = LLMSettings()
        assert not empty.is_configured()

        configured = LLMSettings(provider="openai")
        assert configured.is_configured()


class TestProjectConfigWithLLM:
    """Tests for ProjectConfig with LLM settings"""

    def test_config_with_llm(self):
        """Test config with LLM settings"""
        from intentlog.storage import ProjectConfig, LLMSettings

        config = ProjectConfig(
            project_name="test",
            llm=LLMSettings(provider="openai", model="gpt-4"),
        )

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"

    def test_config_to_dict_with_llm(self):
        """Test serializing config with LLM"""
        from intentlog.storage import ProjectConfig, LLMSettings

        config = ProjectConfig(
            project_name="test",
            llm=LLMSettings(provider="anthropic"),
        )
        data = config.to_dict()

        assert "llm" in data
        assert data["llm"]["provider"] == "anthropic"

    def test_config_from_dict_with_llm(self):
        """Test deserializing config with LLM"""
        from intentlog.storage import ProjectConfig

        data = {
            "project_name": "test",
            "llm": {
                "provider": "ollama",
                "model": "mistral",
            },
        }
        config = ProjectConfig.from_dict(data)

        assert config.llm.provider == "ollama"
        assert config.llm.model == "mistral"

    def test_config_from_dict_without_llm(self):
        """Test deserializing config without LLM section"""
        from intentlog.storage import ProjectConfig

        data = {
            "project_name": "test",
        }
        config = ProjectConfig.from_dict(data)

        assert not config.llm.is_configured()

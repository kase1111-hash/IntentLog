"""
LLM Provider Interface

Defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMError(Exception):
    """Base exception for LLM errors"""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Raised when authentication fails"""
    pass


class ModelNotFoundError(LLMError):
    """Raised when specified model is not available"""
    pass


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: str  # "openai", "anthropic", "ollama", "local"
    model: str = ""  # Model name/ID
    api_key: Optional[str] = None  # API key (or env var name)
    api_key_env: Optional[str] = None  # Environment variable for API key
    base_url: Optional[str] = None  # Custom API endpoint
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment"""
        import os
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class LLMResponse:
    """Response from an LLM completion request"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)  # tokens used
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding request"""
    embedding: List[float]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - complete(): Generate text completion
    - embed(): Generate text embedding (optional)
    - is_available(): Check if provider is configured
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier"""
        pass

    @property
    def default_model(self) -> str:
        """Default model for this provider"""
        return ""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """
        Generate a text completion.

        Args:
            prompt: The user prompt/question
            system: Optional system prompt for context

        Returns:
            LLMResponse with generated content
        """
        pass

    def embed(self, text: str) -> EmbeddingResponse:
        """
        Generate an embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResponse with embedding vector

        Raises:
            NotImplementedError: If provider doesn't support embeddings
        """
        raise NotImplementedError(f"{self.name} does not support embeddings")

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts.

        Default implementation calls embed() for each text.
        Providers may override for batch efficiency.
        """
        return [self.embed(text) for text in texts]

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is properly configured and available.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    def get_model(self) -> str:
        """Get the model to use (config or default)"""
        return self.config.model or self.default_model

    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.get_model():
            errors.append(f"No model specified for {self.name}")
        return errors


class MockProvider(LLMProvider):
    """
    Mock provider for testing.

    Returns predictable responses without making API calls.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(provider="mock", model="mock-model")
        super().__init__(config)
        self._responses: List[str] = []
        self._embeddings: List[List[float]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def set_responses(self, responses: List[str]) -> None:
        """Set responses to return in order"""
        self._responses = list(responses)

    def set_embeddings(self, embeddings: List[List[float]]) -> None:
        """Set embeddings to return in order"""
        self._embeddings = list(embeddings)

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = f"Mock response to: {prompt[:50]}..."

        return LLMResponse(
            content=content,
            model=self.get_model(),
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())},
            finish_reason="stop",
        )

    def embed(self, text: str) -> EmbeddingResponse:
        if self._embeddings:
            embedding = self._embeddings.pop(0)
        else:
            # Generate deterministic mock embedding based on text
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:128]]

        return EmbeddingResponse(
            embedding=embedding,
            model=self.get_model(),
            usage={"total_tokens": len(text.split())},
        )

    def is_available(self) -> bool:
        return True

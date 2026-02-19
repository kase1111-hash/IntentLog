"""
LLM Provider Interface

Defines the abstract interface that all LLM providers must implement.
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMError(Exception):
    """Base exception for LLM errors"""
    pass


class SecretLeakageError(LLMError):
    """Raised when outbound content contains potential secrets"""
    pass


# Patterns for detecting secrets in outbound content
_SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    (re.compile(r"(?:sk|pk|api)[_-]?(?:live|test|key)?[_-]?[a-zA-Z0-9]{20,60}"), "api_key"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9._\-]{20,}"), "bearer_token"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "private_key_pem"),
    (re.compile(r"://[^:\s]+:[^@\s]+@"), "password_in_url"),
    (re.compile(r"gh[ps]_[a-zA-Z0-9]{36,}"), "github_token"),
    (re.compile(r"xox[bpras]-[a-zA-Z0-9\-]{10,}"), "slack_token"),
    (re.compile(r"eyJ[a-zA-Z0-9_-]{20,}\.eyJ[a-zA-Z0-9_-]{20,}"), "jwt_token"),
]


def scan_for_secrets(text: str) -> List[str]:
    """
    Scan text for common secret/credential patterns.

    Args:
        text: The text to scan

    Returns:
        List of detected pattern names (empty if clean)
    """
    if not text:
        return []

    detected = []
    for pattern, name in _SECRET_PATTERNS:
        if pattern.search(text):
            if name not in detected:
                detected.append(name)
    return detected


def check_outbound_content(*texts: Optional[str]) -> None:
    """
    Check outbound content for secrets before sending to LLM APIs.

    Args:
        *texts: Text content to scan

    Raises:
        SecretLeakageError: If secrets detected and override not set
    """
    if os.environ.get("INTENTLOG_ALLOW_SECRETS", "").strip() == "1":
        return

    all_detected = []
    for text in texts:
        if text:
            all_detected.extend(scan_for_secrets(text))

    if all_detected:
        unique = sorted(set(all_detected))
        raise SecretLeakageError(
            f"Outbound content contains potential secrets: {', '.join(unique)}. "
            "Set INTENTLOG_ALLOW_SECRETS=1 to override."
        )


# ---- Known LLM API base URLs (for anomaly detection) ----
KNOWN_API_DOMAINS = {
    "api.openai.com",
    "api.anthropic.com",
    "localhost",
    "127.0.0.1",
}


class CommunicationMonitor:
    """
    Monitors outbound communication to LLM APIs for anomaly detection.

    Tracks request count, destinations, and payload sizes per session.
    Logs warnings when anomalous patterns are detected.
    """

    def __init__(self):
        self._requests: list = []
        self._request_count = 0
        self._start_time = 0.0

    def record_request(
        self,
        url: str,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        status_code: int = 0,
        duration_ms: float = 0.0,
    ) -> Optional[str]:
        """
        Record an outbound request and check for anomalies.

        Returns:
            Warning message if anomaly detected, else None
        """
        import time
        from urllib.parse import urlparse

        now = time.time()
        if self._start_time == 0.0:
            self._start_time = now

        self._request_count += 1
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        entry = {
            "timestamp": now,
            "url": url,
            "hostname": hostname,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }
        self._requests.append(entry)

        # --- Anomaly checks ---
        warnings = []

        # Check for unknown domains
        if hostname and hostname not in KNOWN_API_DOMAINS:
            is_known = False
            for known in KNOWN_API_DOMAINS:
                if hostname.endswith("." + known):
                    is_known = True
                    break
            if not is_known:
                warnings.append(f"Request to unknown API domain: {hostname}")

        # Check for unusual request frequency (> 100/min)
        elapsed = now - self._start_time
        if elapsed > 0 and self._request_count > 100:
            rate = self._request_count / (elapsed / 60.0)
            if rate > 100:
                warnings.append(f"High request rate: {rate:.0f} requests/min")

        # Check for large payloads (> 1MB)
        if bytes_sent > 1_000_000:
            warnings.append(f"Large outbound payload: {bytes_sent} bytes")

        return "; ".join(warnings) if warnings else None

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for this session."""
        if not self._requests:
            return {"total_requests": 0}

        hostnames = set(r["hostname"] for r in self._requests)
        total_sent = sum(r["bytes_sent"] for r in self._requests)
        total_received = sum(r["bytes_received"] for r in self._requests)

        return {
            "total_requests": self._request_count,
            "unique_hosts": sorted(hostnames),
            "total_bytes_sent": total_sent,
            "total_bytes_received": total_received,
        }


# Global communication monitor instance
_comm_monitor: Optional["CommunicationMonitor"] = None


def get_communication_monitor() -> CommunicationMonitor:
    """Get the global communication monitor singleton."""
    global _comm_monitor
    if _comm_monitor is None:
        _comm_monitor = CommunicationMonitor()
    return _comm_monitor


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

    def __post_init__(self):
        """Validate config after initialization."""
        if self.base_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname or ""
            is_known = hostname in KNOWN_API_DOMAINS
            if not is_known:
                for known in KNOWN_API_DOMAINS:
                    if hostname.endswith("." + known):
                        is_known = True
                        break
            if not is_known:
                import warnings
                warnings.warn(
                    f"LLM base_url points to unknown domain: {hostname}. "
                    f"Known domains: {', '.join(sorted(KNOWN_API_DOMAINS))}",
                    UserWarning,
                    stacklevel=2,
                )

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment"""
        if self.api_key:
            import warnings
            warnings.warn(
                "Passing api_key directly in LLMConfig is deprecated and insecure. "
                "Use api_key_env to reference an environment variable instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class LLMResponse:
    """Response from an LLM completion request.

    The ``tainted`` field marks this response as untrusted external data.
    Callers should never interpret response content as executable instructions.
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)  # tokens used
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    tainted: bool = True  # Response is untrusted external data


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

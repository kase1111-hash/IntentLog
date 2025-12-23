"""
OpenAI Provider for IntentLog

Provides integration with OpenAI's API for completions and embeddings.
"""

import json
from typing import Optional, List, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .provider import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    EmbeddingResponse,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Supports GPT-4, GPT-3.5-turbo, and embedding models.
    Uses urllib for HTTP requests to avoid external dependencies.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _get_base_url(self) -> str:
        return self.config.base_url or self.DEFAULT_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        api_key = self.config.get_api_key()
        if not api_key:
            raise AuthenticationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable "
                "or provide api_key in config."
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenAI API"""
        url = f"{self._get_base_url()}/{endpoint}"

        try:
            request = Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=self._get_headers(),
                method="POST",
            )

            with urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            try:
                error_data = json.loads(body)
                error_msg = error_data.get("error", {}).get("message", body)
            except json.JSONDecodeError:
                error_msg = body

            if e.code == 401:
                raise AuthenticationError(f"Authentication failed: {error_msg}")
            elif e.code == 429:
                retry_after = e.headers.get("Retry-After")
                raise RateLimitError(
                    f"Rate limit exceeded: {error_msg}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            elif e.code == 404:
                raise ModelNotFoundError(f"Model not found: {error_msg}")
            else:
                raise LLMError(f"OpenAI API error ({e.code}): {error_msg}")

        except URLError as e:
            raise LLMError(f"Network error: {e.reason}")

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate completion using OpenAI chat API"""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.get_model(),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = self._make_request("chat/completions", data)

        choice = response["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=response["model"],
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
            raw_response=response,
        )

    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding using OpenAI embeddings API"""
        embedding_model = self.config.extra.get(
            "embedding_model", self.DEFAULT_EMBEDDING_MODEL
        )

        data = {
            "model": embedding_model,
            "input": text,
        }

        response = self._make_request("embeddings", data)

        return EmbeddingResponse(
            embedding=response["data"][0]["embedding"],
            model=response["model"],
            usage=response.get("usage", {}),
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts in one request"""
        if not texts:
            return []

        embedding_model = self.config.extra.get(
            "embedding_model", self.DEFAULT_EMBEDDING_MODEL
        )

        data = {
            "model": embedding_model,
            "input": texts,
        }

        response = self._make_request("embeddings", data)

        results = []
        for item in response["data"]:
            results.append(
                EmbeddingResponse(
                    embedding=item["embedding"],
                    model=response["model"],
                    usage=response.get("usage", {}),
                )
            )

        return results

    def is_available(self) -> bool:
        """Check if OpenAI is configured with valid API key"""
        api_key = self.config.get_api_key()
        return api_key is not None and len(api_key) > 0

    def validate_config(self) -> List[str]:
        """Validate OpenAI configuration"""
        errors = super().validate_config()

        if not self.config.get_api_key():
            errors.append(
                "OpenAI API key not configured. Set OPENAI_API_KEY or provide api_key."
            )

        return errors

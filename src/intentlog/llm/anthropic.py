"""
Anthropic Provider for IntentLog

Provides integration with Anthropic's Claude API for completions.
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


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Supports Claude 3.5, Claude 3, and other Claude models.
    Uses urllib for HTTP requests to avoid external dependencies.
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _get_base_url(self) -> str:
        return self.config.base_url or self.DEFAULT_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        api_key = self.config.get_api_key()
        if not api_key:
            raise AuthenticationError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable "
                "or provide api_key in config."
            )
        return {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API"""
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
                error_type = error_data.get("error", {}).get("type", "")
            except json.JSONDecodeError:
                error_msg = body
                error_type = ""

            if e.code == 401:
                raise AuthenticationError(f"Authentication failed: {error_msg}")
            elif e.code == 429:
                retry_after = e.headers.get("Retry-After")
                raise RateLimitError(
                    f"Rate limit exceeded: {error_msg}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            elif e.code == 404 or error_type == "not_found_error":
                raise ModelNotFoundError(f"Model not found: {error_msg}")
            else:
                raise LLMError(f"Anthropic API error ({e.code}): {error_msg}")

        except URLError as e:
            raise LLMError(f"Network error: {e.reason}")

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate completion using Anthropic messages API"""
        data = {
            "model": self.get_model(),
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            data["system"] = system

        # Add temperature if not default
        if self.config.temperature != 1.0:
            data["temperature"] = self.config.temperature

        response = self._make_request("messages", data)

        # Extract content from response
        content_blocks = response.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        return LLMResponse(
            content=content,
            model=response["model"],
            usage={
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=response.get("stop_reason"),
            raw_response=response,
        )

    def embed(self, text: str) -> EmbeddingResponse:
        """
        Anthropic doesn't provide embeddings API.

        For semantic features, use OpenAI embeddings or a local model.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use OpenAI or a local embedding model instead."
        )

    def is_available(self) -> bool:
        """Check if Anthropic is configured with valid API key"""
        api_key = self.config.get_api_key()
        return api_key is not None and len(api_key) > 0

    def validate_config(self) -> List[str]:
        """Validate Anthropic configuration"""
        errors = super().validate_config()

        if not self.config.get_api_key():
            errors.append(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY or provide api_key."
            )

        return errors

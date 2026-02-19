"""
Anthropic Provider for IntentLog

Provides integration with Anthropic's Claude API for completions.
Includes rate limiting and retry logic for production reliability.
"""

import json
import time as _time
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
    check_outbound_content,
    get_communication_monitor,
)
from ..logging import get_logger


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Supports Claude model families including Claude 4, Claude 3.5, and Claude 3.
    Configure specific models via the model parameter in LLMConfig.
    Uses urllib for HTTP requests to avoid external dependencies.

    Available models (as of 2025):
    - claude-sonnet-4-20250514 (latest Sonnet)
    - claude-opus-4-5-20251101 (latest Opus)
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
    - claude-3-opus-20240229
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
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

    def _make_request_internal(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API (internal, without rate limiting)"""
        url = f"{self._get_base_url()}/{endpoint}"
        logger = get_logger()
        monitor = get_communication_monitor()

        encoded_body = json.dumps(data).encode("utf-8")
        start = _time.perf_counter()

        try:
            request = Request(
                url,
                data=encoded_body,
                headers=self._get_headers(),
                method="POST",
            )

            with urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read()
                status_code = response.status
                result = json.loads(raw.decode("utf-8"))

            duration_ms = (_time.perf_counter() - start) * 1000
            logger.info(
                "Anthropic API request",
                endpoint=endpoint, status=status_code,
                duration_ms=round(duration_ms, 1),
            )
            anomaly = monitor.record_request(
                url, bytes_sent=len(encoded_body),
                bytes_received=len(raw),
                status_code=status_code, duration_ms=duration_ms,
            )
            if anomaly:
                logger.warning("Communication anomaly: %s", anomaly)

            return result

        except HTTPError as e:
            duration_ms = (_time.perf_counter() - start) * 1000
            body = e.read().decode("utf-8") if e.fp else ""
            monitor.record_request(
                url, bytes_sent=len(encoded_body),
                status_code=e.code, duration_ms=duration_ms,
            )
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
                logger.warning(
                    "Anthropic rate limit hit",
                    endpoint=endpoint,
                    retry_after=retry_after
                )
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

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API"""
        return self._make_request_internal(endpoint, data)

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate completion using Anthropic messages API"""
        check_outbound_content(prompt, system)

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

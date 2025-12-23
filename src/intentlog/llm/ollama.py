"""
Ollama Provider for IntentLog

Provides integration with Ollama for local LLM inference.
Supports both completion and embedding models.
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
    ModelNotFoundError,
)


class OllamaProvider(LLMProvider):
    """
    Ollama local LLM provider.

    Supports local models like Llama, Mistral, Phi, etc.
    Requires Ollama to be running locally.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _get_base_url(self) -> str:
        return self.config.base_url or self.DEFAULT_BASE_URL

    def _make_request(
        self, endpoint: str, data: Dict[str, Any], stream: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama API"""
        url = f"{self._get_base_url()}/{endpoint}"

        try:
            request = Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(request, timeout=self.config.timeout) as response:
                if stream:
                    # For streaming, read all chunks and combine
                    full_response = ""
                    for line in response:
                        chunk = json.loads(line.decode("utf-8"))
                        if "response" in chunk:
                            full_response += chunk["response"]
                        if chunk.get("done"):
                            return {
                                "response": full_response,
                                "model": chunk.get("model", data.get("model")),
                                "done": True,
                                "total_duration": chunk.get("total_duration"),
                                "eval_count": chunk.get("eval_count", 0),
                                "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                            }
                    return {"response": full_response, "model": data.get("model")}
                else:
                    return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            try:
                error_data = json.loads(body)
                error_msg = error_data.get("error", body)
            except json.JSONDecodeError:
                error_msg = body

            if e.code == 404:
                raise ModelNotFoundError(
                    f"Model '{data.get('model')}' not found. "
                    f"Run 'ollama pull {data.get('model')}' first. Error: {error_msg}"
                )
            else:
                raise LLMError(f"Ollama API error ({e.code}): {error_msg}")

        except URLError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self._get_base_url()}. "
                f"Is Ollama running? Error: {e.reason}"
            )

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate completion using Ollama generate API"""
        data = {
            "model": self.get_model(),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if system:
            data["system"] = system

        response = self._make_request("api/generate", data)

        return LLMResponse(
            content=response.get("response", ""),
            model=response.get("model", self.get_model()),
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            },
            finish_reason="stop" if response.get("done") else None,
            raw_response=response,
        )

    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding using Ollama embeddings API"""
        embedding_model = self.config.extra.get(
            "embedding_model", self.DEFAULT_EMBEDDING_MODEL
        )

        data = {
            "model": embedding_model,
            "prompt": text,
        }

        response = self._make_request("api/embeddings", data)

        return EmbeddingResponse(
            embedding=response.get("embedding", []),
            model=response.get("model", embedding_model),
            usage={},
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts.

        Ollama doesn't support batch embeddings, so we call embed() for each.
        """
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            url = f"{self._get_base_url()}/api/tags"
            request = Request(url, method="GET")
            with urlopen(request, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            url = f"{self._get_base_url()}/api/tags"
            request = Request(url, method="GET")
            with urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []

    def validate_config(self) -> List[str]:
        """Validate Ollama configuration"""
        errors = super().validate_config()

        if not self.is_available():
            errors.append(
                f"Ollama not available at {self._get_base_url()}. "
                "Make sure Ollama is running."
            )

        return errors

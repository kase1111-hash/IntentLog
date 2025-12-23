"""
LLM Provider Registry

Manages registration and retrieval of LLM providers.
"""

from typing import Dict, Type, List, Optional
from .provider import LLMProvider, LLMConfig, LLMError


# Registry of provider classes
_providers: Dict[str, Type[LLMProvider]] = {}


def register_provider(name: str, provider_class: Type[LLMProvider]) -> None:
    """
    Register an LLM provider.

    Args:
        name: Provider identifier (e.g., "openai", "anthropic")
        provider_class: The provider class to register
    """
    _providers[name.lower()] = provider_class


def get_provider(config: LLMConfig) -> LLMProvider:
    """
    Get an instance of the specified provider.

    Args:
        config: LLM configuration with provider name

    Returns:
        Configured LLMProvider instance

    Raises:
        LLMError: If provider is not registered
    """
    name = config.provider.lower()

    if name not in _providers:
        # Try to auto-register known providers
        _auto_register_providers()

    if name not in _providers:
        available = ", ".join(_providers.keys()) if _providers else "none"
        raise LLMError(
            f"Unknown LLM provider: '{name}'. Available providers: {available}"
        )

    provider_class = _providers[name]
    return provider_class(config)


def list_providers() -> List[str]:
    """List all registered provider names"""
    _auto_register_providers()
    return list(_providers.keys())


def _auto_register_providers() -> None:
    """Auto-register built-in providers if not already registered"""
    from .provider import MockProvider

    if "mock" not in _providers:
        register_provider("mock", MockProvider)

    # Try to import and register optional providers
    try:
        from .openai import OpenAIProvider
        if "openai" not in _providers:
            register_provider("openai", OpenAIProvider)
    except ImportError:
        pass

    try:
        from .anthropic import AnthropicProvider
        if "anthropic" not in _providers:
            register_provider("anthropic", AnthropicProvider)
    except ImportError:
        pass

    try:
        from .ollama import OllamaProvider
        if "ollama" not in _providers:
            register_provider("ollama", OllamaProvider)
    except ImportError:
        pass


def get_available_providers() -> List[str]:
    """Get list of providers that are currently available (configured and working)"""
    _auto_register_providers()
    available = []

    for name, provider_class in _providers.items():
        try:
            # Create with minimal config to check availability
            config = LLMConfig(provider=name)
            provider = provider_class(config)
            if provider.is_available():
                available.append(name)
        except Exception:
            pass

    return available

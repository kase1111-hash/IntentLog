# API Reference

This section provides detailed API documentation for all IntentLog modules.

## Module Overview

| Module | Description |
|--------|-------------|
| [`intentlog.core`](core.md) | Core data models (Intent, IntentLog) |
| [`intentlog.storage`](storage.md) | Persistent storage with Merkle chains |
| [`intentlog.cli`](cli.md) | Command-line interface |
| [`intentlog.crypto`](crypto.md) | Ed25519 signing and key management |
| [`intentlog.privacy`](privacy.md) | MP-02 Section 12 encryption and revocation |
| [`intentlog.analytics`](analytics.md) | Intent analytics and metrics |
| [`intentlog.semantic`](semantic.md) | LLM-powered semantic features |
| [`intentlog.mp02`](mp02.md) | MP-02 effort receipt protocol |

## Installation for Development

```bash
# Install with all optional dependencies
pip install intentlog[all]

# Or install specific extras
pip install intentlog[crypto]    # Cryptographic features
pip install intentlog[openai]    # OpenAI LLM provider
pip install intentlog[anthropic] # Anthropic LLM provider
```

## Basic Usage

```python
from intentlog.core import Intent, IntentLog
from intentlog.storage import IntentLogStorage

# Initialize storage
storage = IntentLogStorage()
storage.init_project("my-project")

# Create an intent
intent = storage.add_intent(
    name="Add user authentication",
    reasoning="Implementing JWT-based auth to secure API endpoints",
    metadata={"priority": "high"}
)

# Load and search intents
intents = storage.load_intents()
results = storage.search_intents("authentication")
```

## Architecture

IntentLog follows a modular architecture:

```
intentlog/
├── core.py          # Data models
├── storage.py       # Persistence layer
├── cli/             # CLI commands
├── crypto.py        # Cryptographic operations
├── privacy.py       # Privacy controls
├── analytics.py     # Analytics engine
├── metrics.py       # Doctrine metrics
├── semantic.py      # LLM integration
├── llm/             # LLM providers
│   ├── provider.py  # Abstract interface
│   ├── openai.py    # OpenAI implementation
│   ├── anthropic.py # Anthropic implementation
│   └── ollama.py    # Ollama implementation
└── mp02/            # MP-02 protocol
    ├── observer.py  # Signal capture
    ├── signal.py    # Raw signals
    ├── receipt.py   # Effort receipts
    └── ledger.py    # Append-only ledger
```

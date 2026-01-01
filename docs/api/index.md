# API Reference

This section provides detailed API documentation for all IntentLog modules.

## Module Overview

### Core Modules

| Module | Description |
|--------|-------------|
| [`intentlog.core`](core.md) | Core data models (Intent, IntentLog) |
| [`intentlog.storage`](storage.md) | Persistent storage with Merkle chains |
| [`intentlog.merkle`](storage.md#chainedintent) | Merkle tree chain linking |
| [`intentlog.cli`](cli.md) | Command-line interface (modular) |

### Cryptographic Modules

| Module | Description |
|--------|-------------|
| [`intentlog.crypto`](crypto.md) | Ed25519 signing and key management |
| [`intentlog.privacy`](privacy.md) | MP-02 Section 12 encryption and revocation |

### Analytics Modules

| Module | Description |
|--------|-------------|
| [`intentlog.analytics`](analytics.md) | Intent analytics and statistics |
| [`intentlog.metrics`](analytics.md#intentmetrics) | Doctrine metrics computation |
| [`intentlog.export`](analytics.md#intentexporter) | Multi-format export |
| [`intentlog.sufficiency`](analytics.md#sufficiencytest) | Intent quality testing |

### Context & Tracing Modules

| Module | Description |
|--------|-------------|
| `intentlog.context` | Intent context propagation and session management |
| `intentlog.decorator` | `@intent_logger` decorator for automatic tracing |
| `intentlog.triggers` | Human-in-the-loop trigger system |

### Semantic Modules

| Module | Description |
|--------|-------------|
| [`intentlog.semantic`](semantic.md) | LLM-powered semantic features |
| `intentlog.llm.provider` | Abstract LLM provider interface |
| `intentlog.llm.openai` | OpenAI provider |
| `intentlog.llm.anthropic` | Anthropic Claude provider |
| `intentlog.llm.ollama` | Ollama local models provider |
| `intentlog.llm.registry` | Provider registration and discovery |

### MP-02 Protocol Modules

| Module | Description |
|--------|-------------|
| [`intentlog.mp02`](mp02.md) | MP-02 effort receipt protocol |
| `intentlog.mp02.signal` | Raw signal definitions |
| `intentlog.mp02.observer` | Signal capture |
| `intentlog.mp02.segmentation` | Temporal grouping |
| `intentlog.mp02.validator` | LLM-assisted validation |
| `intentlog.mp02.receipt` | Receipt generation |
| `intentlog.mp02.ledger` | Append-only ledger |

### Integration Modules

| Module | Description |
|--------|-------------|
| `intentlog.integrations.memory_vault` | Memory Vault integration |
| `intentlog.integrations.llm_classifier` | LLM-powered classification |

## Installation

```bash
# Install with all optional dependencies
pip install intentlog[all]

# Or install specific extras
pip install intentlog[crypto]    # Cryptographic features
pip install intentlog[openai]    # OpenAI LLM provider
pip install intentlog[anthropic] # Anthropic LLM provider
```

## Basic Usage

### Core Operations

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

### Context & Decorator

```python
from intentlog.decorator import intent_logger, LogLevel
from intentlog.context import session_scope, get_current_intent

# Automatic function logging
@intent_logger(
    name="process_order",
    reasoning="Order processing with validation",
    level=LogLevel.INFO
)
def process_order(order_id: str):
    current = get_current_intent()
    print(f"Processing in context: {current.name}")

# Session grouping
with session_scope("checkout_flow") as session:
    process_order("order-123")
```

### Human-in-the-Loop Triggers

```python
from intentlog.triggers import (
    requires_approval,
    SensitivityLevel,
    set_trigger_handler,
    ConsoleTriggerHandler
)

set_trigger_handler(ConsoleTriggerHandler())

@requires_approval(
    operation="delete_data",
    sensitivity=SensitivityLevel.CRITICAL
)
def delete_user_data(user_id: str):
    # Will prompt for approval before executing
    pass
```

### Privacy Controls

```python
from intentlog.privacy import PrivacyManager, PrivacyLevel

privacy = PrivacyManager(project_path)

# Encrypt intent
encrypted = privacy.encrypt_intent(intent, level=PrivacyLevel.CONFIDENTIAL)

# Revoke future observation
privacy.revoke_future_observation(user_id="user-123", reason="Privacy request")
```

### Semantic Features

```python
from intentlog.semantic import SemanticEngine, FormalizationType
from intentlog.llm.registry import get_provider

provider = get_provider("openai")
engine = SemanticEngine(provider)

# Semantic search
results = engine.semantic_search(intents, "authentication")

# Formalize intent to code
output = engine.formalize(
    intent,
    formalization_type=FormalizationType.CODE,
    language="python"
)
```

## Architecture

IntentLog follows a modular architecture:

```
intentlog/
├── core.py              # Data models
├── storage.py           # Persistence layer
├── merkle.py            # Hash chain linking
├── crypto.py            # Cryptographic operations
├── privacy.py           # Privacy controls
├── context.py           # Context propagation
├── decorator.py         # @intent_logger
├── triggers.py          # Human-in-the-loop
├── analytics.py         # Analytics engine
├── metrics.py           # Doctrine metrics
├── export.py            # Multi-format export
├── sufficiency.py       # Quality testing
├── semantic.py          # LLM integration
├── audit.py             # Audit logging
├── cli/                 # CLI commands
│   ├── core.py          # Core commands
│   ├── mp02.py          # MP-02 commands
│   ├── analytics.py     # Analytics commands
│   ├── crypto.py        # Crypto commands
│   ├── privacy.py       # Privacy commands
│   └── formalize.py     # Formalization commands
├── llm/                 # LLM providers
│   ├── provider.py      # Abstract interface
│   ├── openai.py        # OpenAI implementation
│   ├── anthropic.py     # Anthropic implementation
│   ├── ollama.py        # Ollama implementation
│   └── registry.py      # Provider registry
├── mp02/                # MP-02 protocol
│   ├── signal.py        # Raw signals
│   ├── observer.py      # Signal capture
│   ├── segmentation.py  # Temporal grouping
│   ├── validator.py     # LLM validation
│   ├── receipt.py       # Effort receipts
│   └── ledger.py        # Append-only ledger
└── integrations/        # External integrations
    ├── memory_vault.py  # Memory Vault
    └── llm_classifier.py# Classification
```

## Exported Symbols

The package exports 328 symbols. Key exports include:

### Core
- `Intent`, `IntentLog`
- `IntentLogStorage`, `ProjectConfig`, `LLMSettings`
- `ChainedIntent`, `MerkleChain`, `verify_chain`

### Cryptographic
- `KeyManager`, `KeyPair`, `Signature`
- `generate_key_pair`, `sign_data`, `verify_signature`
- `CRYPTO_AVAILABLE`

### Privacy
- `PrivacyManager`, `PrivacyLevel`, `AccessPolicy`
- `IntentEncryptor`, `RevocationManager`
- `ENCRYPTION_AVAILABLE`

### Analytics
- `IntentAnalytics`, `IntentMetrics`
- `IntentExporter`, `ExportFormat`
- `SufficiencyTest`, `run_sufficiency_test`

### Context & Decorator
- `IntentContext`, `IntentContextManager`
- `session_scope`, `intent_scope`
- `intent_logger`, `LogLevel`

### Triggers
- `TriggerType`, `SensitivityLevel`
- `requires_approval`, `requires_confirmation`, `requires_review`
- `ConsoleTriggerHandler`, `CallbackTriggerHandler`

### Semantic
- `SemanticEngine`, `FormalizationType`, `FormalizedOutput`
- `SemanticDiff`, `SemanticSearchResult`

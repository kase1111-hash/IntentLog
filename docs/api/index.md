# API Reference

This section provides detailed API documentation for all IntentLog modules.

## Module Overview

### Core Modules

| Module | Description |
|--------|-------------|
| [`intentlog.core`](core.md) | Core data models (Intent, IntentLog) |
| [`intentlog.storage`](storage.md) | Persistent storage with atomic writes and branch management |
| [`intentlog.git`](#git-integration) | Git detection, context capture, hook management |
| [`intentlog.merkle`](storage.md#chainedintent) | Merkle tree chain linking |

### Cryptographic Modules

| Module | Description |
|--------|-------------|
| [`intentlog.crypto`](crypto.md) | Ed25519 signing and key management |

### Analytics Modules

| Module | Description |
|--------|-------------|
| [`intentlog.analytics`](analytics.md) | Intent analytics and statistics |
| [`intentlog.metrics`](analytics.md#intentmetrics) | Doctrine metrics computation |
| [`intentlog.export`](analytics.md#intentexporter) | Multi-format export |

### Context & Tracing Modules

| Module | Description |
|--------|-------------|
| `intentlog.context` | Intent context propagation and session management |
| `intentlog.decorator` | `@intent_logger` decorator for automatic tracing |

### Semantic Modules

| Module | Description |
|--------|-------------|
| [`intentlog.semantic`](semantic.md) | LLM-powered semantic features |
| `intentlog.llm.provider` | Abstract LLM provider interface |
| `intentlog.llm.openai` | OpenAI provider |
| `intentlog.llm.anthropic` | Anthropic Claude provider |
| `intentlog.llm.ollama` | Ollama local models provider |
| `intentlog.llm.registry` | Provider registration and discovery |

### Infrastructure Modules

| Module | Description |
|--------|-------------|
| `intentlog.audit` | Audit logging |
| `intentlog.logging` | Structured logging |
| `intentlog.validation` | Input validation and path sanitization |
| `intentlog.filelock` | Cross-platform file locking |
| `intentlog.schema` | JSON schema validation |
| `intentlog.backup` | Backup and recovery |

## Installation

```bash
# Install with all optional dependencies
pip install intentlog[all]

# Or install specific extras
pip install intentlog[crypto]    # Cryptographic features
pip install intentlog[openai]    # OpenAI LLM provider
pip install intentlog[anthropic] # Anthropic LLM provider
pip install intentlog[llm]       # All LLM providers
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

```
intentlog/
├── core.py              # Data models
├── storage.py           # Persistence layer
├── git.py               # Git integration
├── merkle.py            # Hash chain linking
├── crypto.py            # Cryptographic operations
├── context.py           # Context propagation
├── decorator.py         # @intent_logger
├── analytics.py         # Analytics engine
├── metrics.py           # Doctrine metrics
├── export.py            # Multi-format export
├── semantic.py          # LLM integration
├── audit.py             # Audit logging
├── logging.py           # Structured logging
├── validation.py        # Input validation
├── filelock.py          # File locking
├── schema.py            # Schema validation
├── backup.py            # Backup/recovery
├── cli/                 # CLI commands
│   ├── core.py          # Core commands
│   ├── analytics.py     # Analytics commands
│   ├── crypto.py        # Crypto commands
│   ├── formalize.py     # Formalization commands
│   └── completion.py    # Shell completion
└── llm/                 # LLM providers
    ├── provider.py      # Abstract interface
    ├── openai.py        # OpenAI implementation
    ├── anthropic.py     # Anthropic implementation
    ├── ollama.py        # Ollama implementation
    └── registry.py      # Provider registry
```

## Key Exported Symbols

### Core
- `Intent`, `IntentLog`
- `IntentLogStorage`, `ProjectConfig`, `LLMSettings`
- `ChainedIntent`, `MerkleChain`, `verify_chain`

### Cryptographic
- `KeyManager`, `KeyPair`, `Signature`
- `generate_key_pair`, `sign_data`, `verify_signature`
- `CRYPTO_AVAILABLE`

### Analytics
- `IntentAnalytics`, `IntentMetrics`
- `IntentExporter`, `ExportFormat`

### Context & Decorator
- `IntentContext`, `IntentContextManager`
- `session_scope`, `intent_scope`
- `intent_logger`, `LogLevel`

### Semantic
- `SemanticEngine`, `FormalizationType`, `FormalizedOutput`
- `SemanticDiff`, `SemanticSearchResult`

### Backup
- `BackupManager`, `BackupMetadata`
- `create_backup`, `restore_backup`, `list_backups`

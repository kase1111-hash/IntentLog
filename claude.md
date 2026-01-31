# Claude Code Guidelines for IntentLog

## Project Overview

IntentLog is a **version control system for human reasoning**—a Git-like tool that captures the "why" behind decisions, not just "what" changed. It preserves decision-making context and narrative history as first-class versioned artifacts.

**Key concepts:**
- **Intent**: A reasoning artifact with unique ID, name, reasoning text, timestamp, and optional parent reference
- **IntentLog**: Manager for adding, retrieving, and searching intents
- **ChainedIntent**: Intent wrapped with Merkle hash-chain linking for cryptographic verification
- **MP-02 Protocol**: Effort-tracking protocol for capturing work signals and generating receipts

## Quick Reference

### Build & Install
```bash
pip install -e ".[all,dev]"      # Full development install
pip install -e ".[crypto]"       # Just cryptographic features
```

### Testing
```bash
pytest tests/ -v                              # Run all tests
pytest tests/ -v --cov=src/intentlog          # With coverage
pytest tests/test_core.py -v                  # Single test file
```

### CLI
```bash
ilog init my-project             # Initialize project
ilog commit "reasoning text"     # Create intent
ilog log                         # View history
ilog search "query"              # Search intents
ilog branch feature-x            # Create branch
ilog chain verify                # Verify chain integrity
```

## Project Structure

```
src/intentlog/
├── core.py          # Intent and IntentLog classes (start here)
├── storage.py       # Persistence, project config, branches
├── merkle.py        # Hash-chain linking and verification
├── crypto.py        # Ed25519 signing, key management
├── semantic.py      # LLM-powered semantic search/diffs
├── context.py       # Intent context propagation
├── decorator.py     # @intent_logger decorator
├── privacy.py       # Encryption, access control
├── triggers.py      # Human-in-the-loop approval
├── analytics.py     # Statistical analysis
├── metrics.py       # Doctrine metrics (Intent Density, etc.)
├── export.py        # JSON, JSONL, CSV, HuggingFace export
├── cli/             # Modular CLI commands
├── mp02/            # MP-02 effort-tracking protocol
├── llm/             # LLM provider plugins (OpenAI, Anthropic, Ollama)
└── integrations/    # External system integrations
```

## Coding Conventions

### Dataclass Pattern
Use dataclasses for all data structures:
```python
@dataclass
class Intent:
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent_name: str = ""
    intent_reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
```

### Exception Hierarchy
Follow the established exception patterns:
- `StorageError` → `ProjectNotFoundError`, `BranchNotFoundError`, etc.
- `CryptoError` → `KeyNotFoundError`, `SignatureError`
- `PrivacyError` → `EncryptionError`, `AccessDeniedError`

### Optional Dependencies
Use the availability pattern for optional features:
```python
try:
    from cryptography.hazmat.primitives import hashes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
```

### Type Hints
All public APIs must have full type hints:
```python
def add_intent(self, intent: Intent) -> ChainedIntent:
    ...
```

### CLI Commands
Register commands using the modular pattern in `cli/`:
```python
def register_*_commands(subparsers: argparse._SubParsersAction) -> None:
    ...
```

### Logging
Use the project's logging module:
```python
from .logging import get_logger
logger = get_logger(__name__)
```

## Key Files to Understand First

1. `src/intentlog/core.py` - Core Intent and IntentLog classes
2. `src/intentlog/storage.py` - How intents are persisted
3. `src/intentlog/merkle.py` - Hash-chain verification
4. `tests/test_core.py` - Usage patterns and expected behaviors

## Storage Structure

Projects store data in `.intentlog/` directory:
```
.intentlog/
├── config.json      # Project metadata
├── intents.json     # All intents + chain metadata
└── branches/        # Per-branch data
```

## Important Patterns

### Intent Chain Model
Each intent has cryptographic linking:
- `intent_hash`: SHA256 of intent content
- `prev_hash`: Previous intent's chain_hash
- `chain_hash`: SHA256(intent_hash + prev_hash)
- Optional Ed25519 signature for authenticity

### Context Propagation
Use Python's context variables for thread-safe tracing:
```python
from .context import IntentContext
with IntentContext(trace_id="...") as ctx:
    ...
```

### LLM Providers
Pluggable architecture with abstract base:
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(...): pass
    @abstractmethod
    async def embed(...): pass
```

## Testing Guidelines

- Follow Arrange-Act-Assert pattern
- Use fixtures for common test data
- Mock external dependencies (LLM providers, file system)
- Each module has corresponding `tests/test_*.py`

## What to Avoid

- Do not introduce dependencies without using the optional dependency pattern
- Do not break the zero-dependency core (basic functionality must work without extras)
- Do not modify chain hash algorithms without updating verification logic
- Ensure Python 3.8+ compatibility (no walrus operators in critical paths)

## Documentation

- API docs in `docs/api/`
- User guides in `docs/guide/`
- Protocol spec in `mp-02-spec.md`
- Philosophy in `Doctrine-of-intent.md`

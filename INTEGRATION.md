# IntentLog Integration Guide

This document describes how IntentLog integrates with other repositories and systems.

## Overview

IntentLog is designed to work as a standalone system or integrate with other tools and frameworks. It follows a modular architecture that allows for seamless integration.

## Integration Points

### 1. Memory Vault Integration

IntentLog can integrate with Memory Vault for secure, classified storage of high-value intent records.

**Status**: Implementation ready (see `src/intentlog/integrations/memory_vault.py`)

**Overview**:
- Low-value intents stored in local memory
- High-value intents (classification >= 2) stored in Memory Vault
- Automatic classification based on intent content
- Full audit trail with tamper-proof Merkle trees

> **For detailed integration instructions, code examples, and classification mapping, see [Memory-Vault-Integration.md](Memory-Vault-Integration.md).**

**Dependencies**:
- Optional: `memory-vault` package

### 2. Git Integration

IntentLog is designed to work alongside Git, not replace it.

**How it works**:
- IntentLog tracks *why* (reasoning, intent)
- Git tracks *what* (code changes)
- Both systems complement each other

**Recommended workflow**:
1. Make code changes
2. Add intent commit: `ilog commit "Explanation of why"`
3. Attach code: `git add . && ilog commit "..." --attach`
4. Regular git commit with reference

### 3. CI/CD Integration

IntentLog includes GitHub Actions workflows for automated auditing.

**Location**: `.github/workflows/intent_audit.yml`

**What it does**:
- Runs on push/PR to main branch
- Validates intent logs for quality
- Fails build if empty reasoning or loops detected

**Setup**:
- Already configured in this repository
- Customize `scripts/audit_intents.py` for your needs

### 4. LLM Integration

IntentLog provides a pluggable LLM provider architecture for semantic features.

**Available Providers**:

| Provider | Module | Features |
|----------|--------|----------|
| OpenAI | `intentlog.llm.openai` | GPT-4, embeddings, semantic search |
| Anthropic | `intentlog.llm.anthropic` | Claude models for validation |
| Ollama | `intentlog.llm.ollama` | Local models, privacy-friendly |
| Mock | `intentlog.llm.provider` | Testing, offline development |

**Configuration**:
```bash
# Configure via CLI
ilog config llm --provider openai --model gpt-4o-mini
ilog config llm --provider anthropic --model claude-sonnet-4-20250514
ilog config llm --provider ollama --model llama2

# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**Python API**:
```python
from intentlog.llm.registry import get_provider
from intentlog.semantic import SemanticEngine

# Get configured provider
provider = get_provider("openai")

# Use semantic engine
engine = SemanticEngine(provider)
results = await engine.semantic_search(intents, "authentication")
diff = await engine.semantic_diff(branch1_intents, branch2_intents)
```

**Features**:
- Semantic search with embeddings
- Semantic diff between branches
- LLM-assisted validation (MP-02)
- Deferred formalization (code generation from prose)
- Intent classification

### 5. Context & Decorator Integration

Use the `@intent_logger` decorator for automatic function tracing.

```python
from intentlog.decorator import intent_logger, LogLevel
from intentlog.context import session_scope

# Automatic logging
@intent_logger(
    name="process_order",
    reasoning="Order processing with validation and payment",
    level=LogLevel.INFO
)
def process_order(order_id: str):
    # Function logic here
    pass

# Session grouping
with session_scope("checkout_flow") as session:
    validate_cart()
    process_payment()
    send_confirmation()
```

**Features**:
- Automatic function entry/exit logging
- Context propagation across nested calls
- Session management for grouping related intents
- Trace IDs and span IDs for distributed tracing
- Environment variable propagation to subprocesses

### 6. Human-in-the-Loop Integration

Use triggers for approval workflows in sensitive operations.

```python
from intentlog.triggers import (
    requires_approval,
    requires_confirmation,
    set_trigger_handler,
    ConsoleTriggerHandler,
    SensitivityLevel
)

# Configure handler
set_trigger_handler(ConsoleTriggerHandler())

# Require approval for sensitive operations
@requires_approval(
    operation="delete_user_data",
    sensitivity=SensitivityLevel.CRITICAL,
    reason="Permanent data deletion requires approval"
)
def delete_user_data(user_id: str):
    # Will prompt for approval before executing
    pass

# Or use confirmation for less critical operations
@requires_confirmation(
    operation="send_email",
    sensitivity=SensitivityLevel.MEDIUM
)
def send_bulk_email(recipients: list):
    pass
```

**Trigger Types**:
- `notification`: Inform without blocking
- `confirmation`: Require yes/no response
- `approval`: Require explicit approval with audit trail
- `review`: Queue for later review

### 7. Privacy Controls Integration

Integrate privacy controls per MP-02 Section 12.

```python
from intentlog.privacy import PrivacyManager, PrivacyLevel

privacy = PrivacyManager(project_path)

# Encrypt sensitive intents
encrypted = privacy.encrypt_intent(intent, level=PrivacyLevel.CONFIDENTIAL)

# Revoke future observation
record = privacy.revoke_future_observation(
    user_id="user-123",
    reason="Privacy request"
)

# Check revocation status
if privacy.is_revoked():
    print("Observation has been revoked")
```

**Privacy Levels**:
- `PUBLIC`: No restrictions
- `INTERNAL`: Organization only
- `CONFIDENTIAL`: Need-to-know basis
- `SECRET`: Highly restricted
- `TOP_SECRET`: Maximum protection

### 8. Other Repository Integrations

IntentLog can integrate with various types of projects:

#### Development Repositories
- **Purpose**: Track architecture decisions, refactoring rationale
- **Integration**: Install IntentLog as dev dependency
- **Use**: `ilog commit` alongside `git commit`

#### Research Repositories
- **Purpose**: Document experimental reasoning, failed hypotheses
- **Integration**: Track reasoning alongside notebooks
- **Use**: Branch for different hypotheses

#### Policy/Governance Repositories
- **Purpose**: Version control for organizational decisions
- **Integration**: Prose-first commits for policy evolution
- **Use**: Mergeable decision rationales

#### Agent/AI Repositories
- **Purpose**: Track instruction evolution, debugging emergent behavior
- **Integration**: Log agent intents during execution
- **Use**: Audit trail for agent decision-making

## Repository Compatibility

### Required from Other Repositories

For IntentLog to integrate with your repository:

1. **Python 3.8+** (for Python-based integration)
2. **Git** (for version control)
3. **Optional**: `cryptography` package (for signing/encryption)
4. **Optional**: LLM API access (for semantic features)
5. **Optional**: Memory Vault (for secure storage)

### What IntentLog Provides

- `intentlog` Python package
- `ilog` / `intentlog` CLI tools
- Audit scripts for quality validation
- Integration modules for external systems
- GitHub Actions workflows
- MkDocs documentation

## Installation

### Basic Installation

```bash
pip install intentlog
```

### With Optional Dependencies

```bash
# Cryptographic features
pip install intentlog[crypto]

# LLM integration
pip install intentlog[openai]
pip install intentlog[anthropic]

# All features
pip install intentlog[all]
```

### From Source

```bash
git clone https://github.com/kase1111-hash/IntentLog
cd IntentLog
pip install -e ".[all]"
```

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `ilog init <project>` | Initialize IntentLog project |
| `ilog commit <message>` | Create intent commit |
| `ilog branch [name]` | Manage branches |
| `ilog log` | View intent history |
| `ilog search <query>` | Search intents |
| `ilog status` | Show project status |
| `ilog diff <branch-spec>` | Semantic diff between branches |
| `ilog merge <source>` | Merge branches with explanation |
| `ilog config <setting>` | Configure settings |

### MP-02 Protocol Commands

| Command | Description |
|---------|-------------|
| `ilog observe <action>` | Manage observation sessions |
| `ilog segment <action>` | Mark segment boundaries |
| `ilog receipt <action>` | Manage effort receipts |
| `ilog ledger <action>` | Manage append-only ledger |
| `ilog verify [target]` | Verify integrity |

### Analytics Commands

| Command | Description |
|---------|-------------|
| `ilog export` | Export intents (JSON, JSONL, CSV, HuggingFace, OpenAI) |
| `ilog analytics <action>` | Generate analytics reports |
| `ilog metrics <action>` | Compute doctrine metrics |
| `ilog sufficiency` | Run Intent Sufficiency Test |

### Cryptographic Commands

| Command | Description |
|---------|-------------|
| `ilog keys <action>` | Manage signing keys |
| `ilog chain <action>` | Manage intent chain |

### Privacy Commands

| Command | Description |
|---------|-------------|
| `ilog privacy status` | View privacy status |
| `ilog privacy revoke` | Revoke observation consent |
| `ilog privacy encrypt` | Encrypt sensitive intents |
| `ilog privacy keys` | Manage encryption keys |

### Formalization Commands

| Command | Description |
|---------|-------------|
| `ilog formalize <action>` | Derive formal outputs from prose |

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Intent` | `intentlog.core` | Single intent record |
| `IntentLog` | `intentlog.core` | Collection of intents |
| `IntentLogStorage` | `intentlog.storage` | Persistent storage with Merkle chains |
| `ChainedIntent` | `intentlog.merkle` | Intent with chain linking |
| `MerkleChain` | `intentlog.merkle` | Merkle tree chain |

### Cryptographic Classes

| Class | Module | Description |
|-------|--------|-------------|
| `KeyManager` | `intentlog.crypto` | Ed25519 key management |
| `KeyPair` | `intentlog.crypto` | Public/private key pair |
| `Signature` | `intentlog.crypto` | Digital signature |

### Privacy Classes

| Class | Module | Description |
|-------|--------|-------------|
| `PrivacyManager` | `intentlog.privacy` | Privacy controls |
| `IntentEncryptor` | `intentlog.privacy` | Fernet encryption |
| `RevocationManager` | `intentlog.privacy` | Revocation management |
| `AccessPolicy` | `intentlog.privacy` | Access control |

### Analytics Classes

| Class | Module | Description |
|-------|--------|-------------|
| `IntentAnalytics` | `intentlog.analytics` | Statistical analysis |
| `IntentMetrics` | `intentlog.metrics` | Doctrine metrics |
| `IntentExporter` | `intentlog.export` | Multi-format export |

### Context & Decorator

| Class/Function | Module | Description |
|----------------|--------|-------------|
| `intent_logger` | `intentlog.decorator` | Automatic logging decorator |
| `IntentContext` | `intentlog.context` | Context management |
| `session_scope` | `intentlog.context` | Session grouping |

### MP-02 Protocol

| Class | Module | Description |
|-------|--------|-------------|
| `Observer` | `intentlog.mp02.observer` | Signal capture |
| `SegmentationEngine` | `intentlog.mp02.segmentation` | Temporal grouping |
| `ReceiptBuilder` | `intentlog.mp02.receipt` | Receipt generation |
| `Ledger` | `intentlog.mp02.ledger` | Append-only ledger |

### Triggers

| Class | Module | Description |
|-------|--------|-------------|
| `TriggerHandler` | `intentlog.triggers` | Base trigger handler |
| `ConsoleTriggerHandler` | `intentlog.triggers` | Console prompts |
| `CallbackTriggerHandler` | `intentlog.triggers` | Custom callbacks |

## Testing Integration

To verify integration with your repository:

```bash
# Run IntentLog tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/intentlog

# Run benchmarks
pytest benchmarks/ --benchmark-only

# Run audit on your logs
python scripts/audit_intents.py your_log_file.log
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure IntentLog is installed in your environment
2. **Crypto not available**: Install with `pip install intentlog[crypto]`
3. **LLM errors**: Check API keys are set correctly
4. **Memory Vault not found**: Install `memory-vault` if needed
5. **Path issues**: Check that scripts are in correct locations

### Support

- GitHub Issues: https://github.com/kase1111-hash/IntentLog/issues
- Documentation: See `docs/` folder and markdown files in repository root

## Future Integration Plans

- Decentralized storage backends
- Hardware (TPM) binding for critical intents
- Multi-language SDKs beyond Python
- IDE plugins for inline intent logging
- Webhook integrations for external systems
- GraphQL API for querying intents

## License

IntentLog is licensed under CC BY-SA 4.0. Integration in your projects is encouraged with attribution.

---

*Last updated: January 2026*

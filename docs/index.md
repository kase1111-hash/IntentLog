# IntentLog

**Version Control for Human Reasoning**

IntentLog is a Git-like system that makes intent, reasoning, and decision-making first-class artifacts. Rather than tracking just *what* changed (code), IntentLog tracks *why* things changed (human reasoning).

## Key Features

### Core (Phase 1)
- **Prose Commits**: Record intent and reasoning in natural language
- **Branch Management**: Explore alternative directions without polluting main history
- **Semantic Search**: Find intents by meaning, not just keywords
- **History Navigation**: View, filter, and traverse intent history

### Cryptographic Integrity (Phase 2)
- **Merkle Tree Chains**: SHA-256 linked chains for tamper-evidence
- **Ed25519 Signatures**: Digital signatures for authenticity
- **Inclusion Proofs**: Cryptographic proof that an intent exists in the chain

### MP-02 Protocol (Phase 3)
- **Signal Observation**: Capture raw effort signals
- **Temporal Segmentation**: Group signals into effort periods
- **Receipt Generation**: Create cryptographic proofs of effort
- **Append-Only Ledger**: Immutable record of all receipts

### Analytics & Metrics (Phase 4)
- **Intent Analytics**: Latency, frequency, trends, bottlenecks
- **Doctrine Metrics**: Intent Density, Information Density, Auditability
- **Multi-Format Export**: JSON, JSONL, CSV, HuggingFace, OpenAI formats

### Context & Tracing (Phase 5)
- **@intent_logger Decorator**: Automatic function tracing
- **Context Propagation**: Track intents across nested calls
- **Session Management**: Group related intents together
- **Distributed Tracing**: Trace IDs and span IDs for observability

### Privacy Controls (Phase 6)
- **Encryption**: Fernet symmetric encryption for sensitive content
- **Privacy Levels**: PUBLIC, INTERNAL, CONFIDENTIAL, SECRET, TOP_SECRET
- **Access Control**: Granular permissions per intent
- **Revocation**: Block future observation per MP-02 Section 12

### Deferred Formalization (Phase 8)
- **LLM-Powered Derivation**: Generate formal outputs from prose
- **Multiple Output Types**: Code, rules, heuristics, schema, config, spec, tests
- **Provenance Tracking**: Full audit trail from prose to formalized output

### Human-in-the-Loop (Phase 9)
- **Trigger Types**: Notification, confirmation, approval, review
- **Sensitivity Levels**: Low, medium, high, critical
- **Decorator-Based**: `@requires_approval`, `@requires_confirmation`
- **Audit Integration**: Full trail of human decisions

### LLM Integration
- **Pluggable Providers**: OpenAI, Anthropic, Ollama, mock
- **Semantic Search**: Embedding-based similarity search
- **Semantic Diff**: Understand conceptual changes between branches
- **Formalization**: Generate code from intent prose

## Quick Start

```bash
# Install IntentLog
pip install intentlog

# Initialize a project
ilog init my-project

# Create your first intent commit
ilog commit "Implementing user authentication to secure API endpoints"

# View intent history
ilog log

# Search by content
ilog search "authentication"

# Configure LLM for semantic features
ilog config llm --provider openai --model gpt-4o-mini
```

## Documentation

### Getting Started
- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)

### User Guides
- [Core Concepts](guide/concepts.md)
- [CLI Reference](guide/cli.md)
- [MP-02 Protocol](guide/mp02.md)

### API Reference
- [API Overview](api/index.md)
- [Core Module](api/core.md)
- [Storage Module](api/storage.md)
- [Crypto Module](api/crypto.md)
- [Privacy Module](api/privacy.md)
- [Analytics Module](api/analytics.md)
- [Semantic Module](api/semantic.md)
- [MP-02 Module](api/mp02.md)
- [CLI Module](api/cli.md)

## Philosophy

IntentLog is built on the [Doctrine of Intent](guide/concepts.md), which establishes:

1. **Intent Density (Di)**: The resolution at which reasoning is captured
2. **Information Density**: The richness of captured context
3. **Auditability**: The ability to trace decisions back to their origins
4. **Fraud Resistance**: Cryptographic guarantees of authenticity

## Project Status

- **Version**: 0.1.0 (Alpha)
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Reference Implementation**: Q1 2026

## License

IntentLog is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

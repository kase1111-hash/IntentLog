# IntentLog

**Version Control for Human Reasoning**

IntentLog runs alongside Git and captures **the reasoning behind your code changes**. Every `ilog commit` records a prose explanation — what you decided, why you decided it, what alternatives you rejected — and links it to your current git branch, commit, and staged files.

## Key Features

### Core CLI
- **Prose Commits**: Record intent and reasoning in natural language, linked to git context
- **Git Integration**: Auto-captures git branch, HEAD hash, and staged files with each intent
- **Branch Management**: Explore alternative reasoning directions without polluting main history
- **Search**: Find past decisions by keyword
- **Blame**: Show intent reasoning alongside git history for any file

### Cryptographic Integrity (optional)
- **Merkle Tree Chains**: SHA-256 linked chains for tamper-evidence
- **Ed25519 Signatures**: Digital signatures for authenticity
- **Inclusion Proofs**: Cryptographic proof that an intent exists in the chain

### Analytics & Metrics
- **Intent Analytics**: Latency, frequency, trends, bottlenecks
- **Doctrine Metrics**: Intent Density, Information Density, Auditability
- **Multi-Format Export**: JSON, JSONL, CSV, HuggingFace, OpenAI formats

### Context & Tracing
- **@intent_logger Decorator**: Automatic function tracing
- **Context Propagation**: Track intents across nested calls
- **Session Management**: Group related intents together

### Semantic Features (optional)
- **Semantic Search**: Embedding-based similarity search via LLM
- **Semantic Diff**: Understand conceptual changes between branches
- **Formalization**: Generate code/rules/specs from prose intent
- **Pluggable Providers**: OpenAI, Anthropic, Ollama (local)

## Quick Start

```bash
# Install IntentLog
pip install intentlog

# Initialize in an existing git repo
cd your-project
ilog init my-project

# Record your first intent
ilog commit "Starting with a monolith because the team is 3 people. Will split services when we hit scaling pain."

# View intent history
ilog log

# Search by content
ilog search "monolith"

# Show project status with git context
ilog status
```

## Documentation

### Getting Started
- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)

### User Guides
- [Core Concepts](guide/concepts.md)
- [CLI Reference](guide/cli.md)

### API Reference
- [API Overview](api/index.md)
- [Core Module](api/core.md)
- [Storage Module](api/storage.md)
- [Crypto Module](api/crypto.md)
- [Analytics Module](api/analytics.md)
- [Semantic Module](api/semantic.md)

## Philosophy

IntentLog is built on the [Doctrine of Intent](guide/concepts.md), which establishes:

1. **Intent Density (Di)**: The resolution at which reasoning is captured
2. **Information Density**: The richness of captured context
3. **Auditability**: The ability to trace decisions back to their origins
4. **Fraud Resistance**: Cryptographic guarantees of authenticity

## Project Status

- **Version**: 0.2.0 (Alpha)
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12

## License

IntentLog is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

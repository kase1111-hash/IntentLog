# IntentLog

**Version Control for Human Reasoning**

IntentLog is a Git-like system that makes intent, reasoning, and decision-making first-class artifacts. Rather than tracking just *what* changed (code), IntentLog tracks *why* things changed (human reasoning).

## Key Features

- **Prose Commits**: Record intent and reasoning in natural language
- **Cryptographic Integrity**: Merkle tree chain linking with Ed25519 signatures
- **MP-02 Protocol**: Effort receipt generation for transparent work tracking
- **LLM Integration**: Semantic search, diff, and formalization
- **Privacy Controls**: Encryption, access control, and revocation

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
```

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [CLI Reference](guide/cli.md)
- [API Reference](api/index.md)

## Philosophy

IntentLog is built on the [Doctrine of Intent](guide/concepts.md), which establishes:

1. **Intent Density (Di)**: The resolution at which reasoning is captured
2. **Information Density**: The richness of captured context
3. **Auditability**: The ability to trace decisions back to their origins
4. **Fraud Resistance**: Cryptographic guarantees of authenticity

## License

IntentLog is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

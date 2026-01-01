# IntentLog: Version Control for Human Reasoning

**What if Git tracked *why*, not just *what*?**

[![CI](https://github.com/kase1111-hash/IntentLog/actions/workflows/intent_audit.yml/badge.svg)](https://github.com/kase1111-hash/IntentLog/actions/workflows/intent_audit.yml)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/kase1111-hash/IntentLog)

> **Note**: This is a conceptual design released for collaborative development. Reference implementation coming Q1 2026. Feedback and prototype contributions welcome.

## Overview

Most version control systems excel at capturing syntactic changes—what lines were added, deleted, or modified—but they remain blind to the human reasoning behind those changes. Decisions scatter across chat threads, emails, meeting notes, and forgotten comments, leaving future contributors to reverse-engineer intent from cryptic diffs and commit messages.

IntentLog introduces **semantic versioning for reasoning itself**: prose commits that preserve narrative context, LLM-powered explanations of changes, and traceable evolution of ideas. By making natural language the first-class artifact, it turns decision-making into an auditable, branchable, and interpretable history.

## The Problem

- **Decision context gets lost**: A refactor happens, but the rationale vanishes into ephemeral Slack threads
- **"Why did we do this?" lives in old chats**: Onboarding becomes archaeological work; debugging feels like guesswork
- **ADRs aren't connected to evolution**: Architecture Decision Records are static documents, disconnected from actual commits

## The Solution

**IntentLog is Git, but for intent.**

| Feature | Description |
|---------|-------------|
| **Prose Commits** | Each commit is a natural-language explanation of *why* something changed, cryptographically signed and timestamped |
| **Branches for Exploration** | Experiment with alternative directions without polluting main history |
| **Semantic Diffs** | LLM-generated readable summaries instead of line-by-line changes |
| **Merges via Explanation** | Conflicts reconciled with prose commits narrating trade-offs and rationale |
| **Deferred Formalization** | Ambiguous intent stays in prose; LLMs derive code, rules, or heuristics on demand |
| **Precedent Trails** | Every commit references prior ones—like case law for your project |

## Features

### Core Capabilities (Phase 1)
- `ilog init` - Initialize IntentLog repository
- `ilog commit` - Record intent with prose description
- `ilog log` - View intent history
- `ilog search` - Search intents by content
- `ilog branch` - Branch management
- `ilog status` - Show project status
- `ilog diff` - Semantic diff between branches
- `ilog merge` - Merge with explanatory context

### Cryptographic Integrity (Phase 2)
- Merkle tree chain linking with SHA-256
- Ed25519 digital signatures for authenticity
- `ilog keys generate` - Generate signing keypairs
- `ilog chain verify` - Verify chain integrity
- Inclusion proofs for individual intents

### MP-02 Protocol (Phase 3)
- Observer system for raw signal capture
- Temporal segmentation of effort signals
- LLM-assisted validation and coherence analysis
- `ilog observe` - Start observation session
- `ilog receipt` - Generate cryptographic effort receipts
- `ilog ledger` - Append-only ledger for tamper-evidence

### Analytics & Metrics (Phase 4)
- Intent analytics: latency, frequency, trends, bottlenecks
- Doctrine metrics: Intent Density, Information Density, Auditability
- `ilog analytics` - View intent analytics
- `ilog metrics` - Compute doctrine metrics
- `ilog export` - Export in JSON, JSONL, CSV, HuggingFace, OpenAI formats
- `ilog sufficiency` - Test intent quality

### Context & Decorator (Phase 5)
- `@intent_logger` decorator for automatic function tracing
- Intent context propagation across nested calls
- Session management for grouping related intents
- Full tracing with trace IDs and span IDs
- Context hooks (on_enter, on_exit)
- Environment variable propagation to subprocesses

### Privacy Controls (Phase 6)
- Fernet symmetric encryption for sensitive content
- Privacy levels: PUBLIC, INTERNAL, CONFIDENTIAL, SECRET, TOP_SECRET
- Access control policies with granular permissions
- `ilog privacy revoke` - Revocation mechanism for future observation
- `ilog privacy encrypt` - Encrypt sensitive intents

### Deferred Formalization (Phase 8)
- LLM-powered derivation from prose intent
- `ilog formalize` - Generate code, rules, heuristics, schema, config, spec, or tests
- Full provenance tracking from source intent to formalized output

### Human-in-the-Loop Triggers (Phase 9)
- Multiple trigger types: notification, confirmation, approval, review
- Sensitivity levels: low, medium, high, critical
- `@requires_approval`, `@requires_confirmation`, `@requires_review` decorators
- Timeout and escalation support
- Full audit trail integration

### LLM Integration
- Pluggable provider architecture
- OpenAI provider (GPT-4, embeddings)
- Anthropic Claude provider
- Ollama provider (local models)
- Semantic search with embeddings
- Semantic diff between branches

## Quick Start

```bash
# Install
pip install intentlog

# Initialize a project
ilog init my-project

# Make your first intent commit
ilog commit "We're starting with a monolithic repo because the team is small and we need fast iteration. We'll revisit splitting services once we hit scaling pain."

# Attach code/files if desired
git add .          # or any files
ilog commit "Adding user authentication module" --attach

# Branch for an experiment
ilog branch experimental-event-sourcing
ilog commit "Exploring event sourcing for better auditability—pros: immutable history; cons: steeper learning curve."

# See semantic history
ilog log            # Shows narrative timeline
ilog diff main..experimental-event-sourcing
# → "The experimental branch introduces event sourcing to preserve full history,
#    trading simplicity for long-term audit benefits."

# Query past reasoning semantically
ilog search "why did we choose monolithic over microservices"
# → Returns relevant commits with context, even if exact words differ

# Merge with explanation
ilog merge main --message "After prototyping, we're sticking with relational model for now—event sourcing adds complexity without immediate payoff. Revisit in Q3."
```

Play with it locally—no blockchain, no network required for solo/team use.

## Installation

### Basic Installation

```bash
pip install intentlog
```

### Optional Dependencies

```bash
# Cryptographic features (Ed25519 signing, Fernet encryption)
pip install intentlog[crypto]

# OpenAI LLM integration
pip install intentlog[openai]

# Anthropic Claude integration
pip install intentlog[anthropic]

# All features
pip install intentlog[all]
```

### Development Installation

```bash
git clone https://github.com/kase1111-hash/IntentLog.git
cd IntentLog
pip install -e ".[all,dev]"
```

## Use Cases

### For Development Teams
- **Architecture decision tracking**: Every major refactor or tech choice gets a living, branchable rationale
- **AI agent instruction evolution**: Track how prompts and system instructions evolve—critical for debugging emergent behavior
- **Open source governance**: Proposals, RFCs, and constitution changes live as mergeable prose histories

### Beyond Code
- **Research notebooks**: Scientific reasoning preserved alongside data/experiments; failed hypotheses kept as branches for negative precedent
- **Policy documents**: Organizational policies evolve with visible debate trails—perfect for DAOs, co-ops, or compliance-heavy teams
- **Creative applications**: Screenwriting teams tracking plot motivations; design systems preserving "why this shade of blue?"; collaborative world-building

## Project Structure

```
IntentLog/
├── src/intentlog/           # Main package (~31,000 LOC)
│   ├── core.py              # Core Intent/IntentLog classes
│   ├── storage.py           # Persistent storage with Merkle chains
│   ├── crypto.py            # Ed25519 signatures & key management
│   ├── merkle.py            # Hash chain linking & verification
│   ├── semantic.py          # LLM-powered semantic features
│   ├── context.py           # Intent context management
│   ├── decorator.py         # @intent_logger decorator
│   ├── privacy.py           # Encryption & access control
│   ├── triggers.py          # Human-in-the-loop system
│   ├── analytics.py         # Statistical analysis
│   ├── metrics.py           # Doctrine metrics
│   ├── export.py            # Multi-format export
│   ├── sufficiency.py       # Intent quality testing
│   ├── cli/                 # Modular CLI package
│   │   ├── core.py          # Core commands
│   │   ├── mp02.py          # MP-02 Protocol commands
│   │   ├── analytics.py     # Analytics commands
│   │   ├── crypto.py        # Cryptographic commands
│   │   ├── privacy.py       # Privacy commands
│   │   └── formalize.py     # Formalization commands
│   ├── mp02/                # MP-02 protocol implementation
│   │   ├── observer.py      # Signal capture
│   │   ├── signal.py        # Raw signal definitions
│   │   ├── segmentation.py  # Temporal grouping
│   │   ├── validator.py     # LLM-assisted validation
│   │   ├── receipt.py       # Receipt generation
│   │   └── ledger.py        # Append-only ledger
│   ├── llm/                 # LLM provider plugins
│   │   ├── openai.py        # OpenAI provider
│   │   ├── anthropic.py     # Anthropic provider
│   │   └── ollama.py        # Ollama provider
│   └── integrations/        # External integrations
│       ├── memory_vault.py  # Memory Vault integration
│       └── llm_classifier.py# LLM-powered classification
├── tests/                   # Comprehensive test suite
├── docs/                    # MkDocs documentation
├── examples/                # Usage examples
├── scripts/                 # Utility scripts
├── benchmarks/              # Performance benchmarks
└── .github/workflows/       # CI/CD workflows
```

## Why This Matters Now

In a world where AI can produce code, text, and designs in seconds, the scarce resource is no longer the artifact—it's the **intentional human signal**. By making prose the commit, IntentLog elevates the actual effort—deliberation, trade-offs, and the "why"—to first-class status.

- **AI systems need interpretable instruction histories**: As we build increasingly capable agents, understanding how their guiding instructions evolved becomes essential for safety, alignment, and debugging
- **Distributed teams need shared context**: Remote and asynchronous work amplifies the cost of lost rationale—IntentLog makes reasoning a durable team asset
- **We're entering an era of human-AI co-reasoning**: Tools that preserve narrative intent will separate robust collaborative systems from brittle ones

Every commit is a breadcrumb of cognition. Every branch is an explored possibility. Every merge is a narrated reconciliation.

## Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start Tutorial](docs/getting-started/quickstart.md)
- [Core Concepts](docs/guide/concepts.md)
- [CLI Reference](docs/guide/cli.md)
- [MP-02 Protocol](docs/guide/mp02.md)
- [API Reference](docs/api/index.md)
- [Doctrine of Intent](Doctrine-of-intent.md)
- [MP-02 Specification](mp-02-spec.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/kase1111-hash/IntentLog.git
cd IntentLog
pip install -e ".[all,dev]"

# Run tests
pytest tests/ -v

# Run benchmarks
pytest benchmarks/ --benchmark-only
```

## License

**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**

You are free to share and adapt, provided you give appropriate credit and share alike.

---

**Open for collaboration. Prototype early 2026. Prior art timestamped December 16, 2025.**

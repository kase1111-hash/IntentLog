# IntentLog: Version Control for Human Reasoning

**What if Git tracked *why*, not just *what*?**

[![CI](https://github.com/kase1111-hash/IntentLog/actions/workflows/intent_audit.yml/badge.svg)](https://github.com/kase1111-hash/IntentLog/actions/workflows/intent_audit.yml)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/kase1111-hash/IntentLog)

## What IntentLog Does

IntentLog runs alongside Git and captures **the reasoning behind your code changes**. Every `ilog commit` records a prose explanation — what you decided, why you decided it, what alternatives you rejected — and links it to your current git branch, commit, and staged files.

Later, when someone asks "why did we choose PostgreSQL?" or "what was the rationale for this refactor?", `ilog search` finds the answer instantly.

```bash
# You're about to commit a database migration
git add .
ilog commit "Switching from SQLite to PostgreSQL because we need concurrent writes across 3 services. Considered MongoDB but relational constraints matter more than schema flexibility here."
git commit -m "Migrate to PostgreSQL"

# Six months later, someone asks why
ilog search "database choice"
# → [a1b2c3d4e5f6] 2026-01-15
#   Switching from SQLite to PostgreSQL because we need concurrent writes...
#   git: main @ 7f8e9d0a1b2c
```

## Quick Start

```bash
# Install
pip install intentlog

# Initialize in an existing git repo
cd your-project
ilog init my-project
#   Git repo: /path/to/your-project   ← auto-detected

# Record your first intent
ilog commit "Starting with a monolith because the team is 3 people. Will split services when we hit scaling pain."

# See your reasoning history
ilog log

# Search past decisions
ilog search "monolith"

# See full details for a specific intent
ilog show a1b2c3d4

# Show project status with git context
ilog status
```

## Features

### Core CLI

| Command | Description |
|---------|-------------|
| `ilog init <name>` | Initialize project (auto-detects git repo) |
| `ilog commit <message>` | Record intent with git context (branch, HEAD, staged files) |
| `ilog log` | View intent history with git context |
| `ilog search <query>` | Search intents by keyword |
| `ilog show <id>` | Display a single intent with full metadata |
| `ilog status` | Project status with git info |
| `ilog branch [name]` | Create/switch/list branches |
| `ilog diff <spec>` | Compare branches |
| `ilog merge <branch>` | Merge intent branches |
| `ilog blame <file>` | Show intent reasoning for a file's git history |
| `ilog hooks install` | Install git hooks (prepare-commit-msg) |
| `ilog config` | Configure settings |

All commands support `--json` for scripting where applicable.

### Git Integration

IntentLog's core feature — every intent automatically captures:

- **Git branch name** at time of commit
- **Git HEAD commit hash** for bidirectional linking
- **Staged files list** so you know what code the reasoning applies to

Use `ilog log --git-commit <hash>` to find intents linked to a specific git commit, or `ilog blame <file>` to see reasoning alongside git history.

### Cryptographic Integrity (optional)

With `pip install intentlog[crypto]`:

- Merkle tree chain linking (SHA-256) for tamper-evident history
- Ed25519 digital signatures for authenticity
- `ilog keys generate` / `ilog chain verify`
- Zero-dependency core — crypto is fully optional

### Semantic Features (optional)

With `pip install intentlog[llm]`:

- `ilog search --semantic` for meaning-based search with LLM embeddings
- `ilog diff` with LLM-generated semantic summaries
- `ilog formalize` to derive code/rules/specs from prose intent
- Pluggable providers: OpenAI, Anthropic, Ollama (local)

## Installation

```bash
# Core (no dependencies)
pip install intentlog

# With crypto (Ed25519 signing, chain verification)
pip install intentlog[crypto]

# With LLM features (semantic search, formalization)
pip install intentlog[llm]

# Everything
pip install intentlog[all]
```

### Development

```bash
git clone https://github.com/kase1111-hash/IntentLog.git
cd IntentLog
pip install -e ".[all,dev]"
pytest tests/ -v
```

## What IntentLog is NOT

- **Not a replacement for Git** — it runs alongside Git, not instead of it
- **Not a documentation generator** — it captures reasoning you write, it doesn't generate it
- **Not an AI tool** — LLM features are optional extras, the core is pure Python with zero dependencies
- **Not a blockchain** — Merkle chains provide tamper-evidence locally, no network required

## Project Structure

```
src/intentlog/
├── core.py        # Intent and IntentLog dataclasses
├── storage.py     # Persistence, branches, atomic writes
├── git.py         # Git detection, context capture, hooks
├── merkle.py      # Hash chain linking and verification
├── audit.py       # Audit logging
├── logging.py     # Structured logging
├── validation.py  # Input validation
├── filelock.py    # Cross-platform file locking
├── schema.py      # JSON schema validation
├── cli/           # Modular CLI
│   ├── core.py    # init, commit, log, search, show, blame, hooks, ...
│   ├── analytics.py
│   ├── crypto.py
│   ├── formalize.py
│   └── completion.py
└── llm/           # Optional LLM providers
    ├── openai.py
    ├── anthropic.py
    └── ollama.py
```

## Why This Matters

In a world where AI can produce code in seconds, the scarce resource is no longer the artifact — it's the **reasoning behind it**. Decisions scatter across Slack threads, meeting notes, and forgotten comments. When someone asks "why did we do this?", the answer is usually "check the old chat."

IntentLog makes reasoning a **durable, searchable, versioned artifact** — linked directly to the code it explains.

## Connected Repositories

IntentLog is part of a broader ecosystem exploring intent preservation and human-AI collaboration. Integrations are maintained as separate packages:

- [**NatLangChain**](https://github.com/kase1111-hash/NatLangChain) — Prose-first blockchain protocol
- [**Agent-OS**](https://github.com/kase1111-hash/Agent-OS) — Natural-language native OS for AI agents
- [**memory-vault**](https://github.com/kase1111-hash/memory-vault) — Sovereign storage for cognitive artifacts
- [**boundary-daemon-**](https://github.com/kase1111-hash/boundary-daemon-) — Trust enforcement layer

## License

**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**

You are free to share and adapt, provided you give appropriate credit and share alike.

---

**Open for collaboration. Prior art timestamped December 16, 2025.**

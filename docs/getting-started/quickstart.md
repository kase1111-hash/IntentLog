# Quick Start

This guide walks you through basic IntentLog usage.

## Initialize a Project

```bash
# Create a new IntentLog project in an existing git repo
cd your-project
ilog init my-project
```

This creates a `.intentlog/` directory with:
- Project configuration (including auto-detected git root)
- Branch storage
- Key storage (for signing)

## Create Intent Commits

Record your reasoning as you work:

```bash
# Simple commit — git context (branch, HEAD, staged files) captured automatically
ilog commit "Implementing user authentication to secure API endpoints"

# Multi-line with detailed reasoning
ilog commit "Add caching layer
Redis-based caching to reduce database load.
Chose Redis over Memcached for persistence and data structures."

# Sign with Ed25519 (requires crypto extra)
ilog commit --sign "Critical security fix"
```

## View History

```bash
# Show recent intents
ilog log

# Show more
ilog log --limit 50

# Show specific branch
ilog log --branch feature

# Filter by git commit hash
ilog log --git-commit 7f8e9d0a

# Machine-readable output
ilog log --json
```

## Show Intent Details

```bash
# Display full metadata for a specific intent
ilog show a1b2c3d4
```

## Search Intents

```bash
# Text search
ilog search "authentication"

# Semantic search (requires LLM config)
ilog search --semantic "user login flow"
```

## Git Integration

```bash
# Show project status with git info
ilog status

# See reasoning for a file's git history
ilog blame src/auth.py

# Install git hooks (prompts for intent on git commit)
ilog hooks install
```

## Branch Management

```bash
# List branches
ilog branch --list

# Create and switch
ilog branch feature

# Switch to existing
ilog branch main
```

## Configure LLM (Optional)

Enable semantic features:

```bash
# Configure OpenAI
ilog config llm --provider openai --model gpt-4o-mini

# Configure Anthropic
ilog config llm --provider anthropic

# Configure local Ollama
ilog config llm --provider ollama --model llama2
```

Set API key:

```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

## Cryptographic Features

```bash
# Generate signing key
ilog keys generate --name my-key

# Verify chain integrity
ilog chain verify

# Get inclusion proof
ilog chain proof --sequence 5
```

## Analytics

```bash
# Summary report
ilog analytics summary

# Doctrine metrics
ilog metrics all

# Export for analysis
ilog export --format jsonl --output intents.jsonl
```

## Next Steps

- [Core Concepts](../guide/concepts.md) - Understand the Doctrine of Intent
- [CLI Reference](../guide/cli.md) - Complete command reference
- [API Reference](../api/index.md) - Python API documentation

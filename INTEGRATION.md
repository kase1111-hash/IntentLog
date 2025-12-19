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
- High-value intents (classification ≥ 2) stored in Memory Vault
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

### 4. LLM/AI Integration

IntentLog is designed to work with AI systems for:
- Semantic diff generation
- Intent summarization
- Search and retrieval
- Quality validation

**Planned features** (Q1 2026):
- LLM-powered semantic diffs
- Automated intent classification
- Intelligent search
- Conflict resolution through LLM reasoning

### 5. Other Repository Integrations

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
3. **Optional**: Memory Vault (for secure storage)
4. **Optional**: LLM API access (for semantic features)

### What IntentLog Provides

- `intentlog` Python package
- `ilog` CLI tool
- Audit scripts for quality validation
- Integration modules for external systems
- GitHub Actions workflows

## Installation

### As a Dependency

```bash
# Install from PyPI (when published)
pip install intentlog

# Install from source
git clone https://github.com/kase1111-hash/IntentLog
cd IntentLog
pip install -e .
```

### As a Submodule

```bash
# In your repository
git submodule add https://github.com/kase1111-hash/IntentLog intentlog
cd intentlog
pip install -e .
```

## API Reference

### Core Classes

- `IntentLog`: Main log manager
- `Intent`: Single intent record
- `MemoryVaultIntegration`: Memory Vault integration layer

### CLI Commands

- `ilog init <project>`: Initialize IntentLog
- `ilog commit <message>`: Create intent commit
- `ilog branch <name>`: Create intent branch
- `ilog log`: View intent history
- `ilog search <query>`: Search intents
- `ilog audit <file>`: Audit intent logs

## Testing Integration

To verify integration with your repository:

```bash
# Run IntentLog tests
pytest tests/

# Run audit on your logs
python scripts/audit_intents.py your_log_file.log
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure IntentLog is installed in your environment
2. **Path issues**: Check that scripts are in correct locations
3. **Memory Vault not found**: Install `memory-vault` if needed

### Support

- GitHub Issues: https://github.com/kase1111-hash/IntentLog/issues
- Documentation: See markdown files in repository root

## Future Integration Plans

- Decentralized storage backends
- Hardware (TPM) binding for critical intents
- Advanced LLM semantic features
- Multi-language support beyond Python
- IDE plugins for inline intent logging

## License

IntentLog is licensed under CC BY-SA 4.0. Integration in your projects is encouraged with attribution.

---

*Last updated: December 2025*

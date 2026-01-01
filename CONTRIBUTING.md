# Contributing to IntentLog

Thank you for your interest in contributing to IntentLog! This document provides guidelines and instructions for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher (3.9-3.12 tested in CI)
- Git
- Basic understanding of version control concepts

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kase1111-hash/IntentLog.git
   cd IntentLog
   ```

2. **Install in development mode with all dependencies**:
   ```bash
   pip install -e ".[all,dev]"
   ```

   Or install specific dependency groups:
   ```bash
   pip install -e ".[crypto]"      # Cryptographic features
   pip install -e ".[openai]"      # OpenAI integration
   pip install -e ".[anthropic]"   # Anthropic integration
   pip install -e ".[docs]"        # Documentation tools
   pip install -e ".[benchmark]"   # Performance testing
   ```

3. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

## Project Structure

```
IntentLog/
├── .github/
│   └── workflows/              # GitHub Actions CI/CD
│       ├── docs.yml            # Documentation build
│       └── intent_audit.yml    # Automated intent validation
├── src/intentlog/              # Main package (~31,000 LOC)
│   ├── __init__.py             # Package exports (328 symbols)
│   ├── core.py                 # Core Intent/IntentLog classes
│   ├── storage.py              # Persistent storage with Merkle chains
│   ├── crypto.py               # Ed25519 signatures & key management
│   ├── merkle.py               # Hash chain linking & verification
│   ├── semantic.py             # LLM-powered semantic features
│   ├── context.py              # Intent context management
│   ├── decorator.py            # @intent_logger decorator
│   ├── privacy.py              # Encryption & access control (MP-02 S12)
│   ├── triggers.py             # Human-in-the-loop system
│   ├── analytics.py            # Statistical analysis
│   ├── metrics.py              # Doctrine metrics
│   ├── export.py               # Multi-format export
│   ├── sufficiency.py          # Intent quality testing
│   ├── audit.py                # Audit logging
│   ├── cli/                    # Modular CLI package
│   │   ├── __init__.py         # CLI entry point
│   │   ├── core.py             # Core commands (init, commit, branch, etc.)
│   │   ├── mp02.py             # MP-02 Protocol commands
│   │   ├── analytics.py        # Analytics commands
│   │   ├── crypto.py           # Cryptographic commands
│   │   ├── privacy.py          # Privacy commands
│   │   ├── formalize.py        # Formalization commands
│   │   └── utils.py            # Shared utilities
│   ├── mp02/                   # MP-02 protocol implementation
│   │   ├── __init__.py
│   │   ├── signal.py           # Raw signal definitions
│   │   ├── observer.py         # Signal capture
│   │   ├── segmentation.py     # Temporal grouping
│   │   ├── validator.py        # LLM-assisted validation
│   │   ├── receipt.py          # Receipt generation
│   │   └── ledger.py           # Append-only ledger
│   ├── llm/                    # LLM provider plugins
│   │   ├── __init__.py
│   │   ├── provider.py         # Abstract interface
│   │   ├── openai.py           # OpenAI provider
│   │   ├── anthropic.py        # Anthropic provider
│   │   ├── ollama.py           # Ollama provider
│   │   └── registry.py         # Provider registration
│   └── integrations/           # External integrations
│       ├── __init__.py
│       ├── memory_vault.py     # Memory Vault integration
│       └── llm_classifier.py   # LLM-powered classification
├── tests/                      # Comprehensive test suite (15 files)
│   ├── test_core.py            # Core functionality tests
│   ├── test_storage.py         # Storage tests
│   ├── test_phase2.py          # Crypto/Merkle tests
│   ├── test_mp02.py            # MP-02 protocol tests
│   ├── test_phase4.py          # Analytics tests
│   ├── test_phase5.py          # Context/decorator tests
│   ├── test_privacy.py         # Privacy tests
│   ├── test_triggers.py        # HITL trigger tests
│   ├── test_formalization.py   # Formalization tests
│   ├── test_llm.py             # LLM provider tests
│   └── ...
├── docs/                       # MkDocs documentation
│   ├── index.md
│   ├── getting-started/
│   ├── guide/
│   └── api/
├── examples/                   # Usage examples
│   └── basic_usage.py
├── scripts/                    # Utility scripts
│   └── audit_intents.py
├── benchmarks/                 # Performance benchmarks
├── pyproject.toml              # Modern Python packaging
└── mkdocs.yml                  # Documentation config
```

## Development Workflow

### Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   pytest tests/ -v

   # With coverage
   pytest tests/ -v --cov=src/intentlog --cov-report=term-missing
   ```

4. **Run benchmarks** (if performance-relevant):
   ```bash
   pytest benchmarks/ --benchmark-only
   ```

5. **Build documentation** (if docs changed):
   ```bash
   mkdocs serve  # Preview at http://127.0.0.1:8000
   ```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Type hints are encouraged for public APIs

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Test both success and failure cases
- Use pytest-asyncio for async tests

Example test structure:
```python
import pytest
from intentlog.core import Intent, IntentLog

def test_intent_creation():
    """Test that Intent can be created with required fields."""
    # Arrange
    name = "Test intent"
    reasoning = "Test reasoning"

    # Act
    intent = Intent(intent_name=name, intent_reasoning=reasoning)

    # Assert
    assert intent.intent_name == name
    assert intent.intent_reasoning == reasoning
    assert intent.intent_id is not None

@pytest.mark.asyncio
async def test_async_feature():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all new code
- Update docs/ for feature documentation
- Update INTEGRATION.md for integration changes
- Create examples in `examples/` for major features
- Update CHANGELOG.md for notable changes

## Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

   Commit message format:
   - Use present tense ("Add feature" not "Added feature")
   - First line: brief summary (50 chars or less)
   - Blank line
   - Detailed description if needed

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Go to GitHub and create a PR
   - Describe your changes in detail
   - Reference any related issues
   - Ensure all CI checks pass

## Types of Contributions

### Bug Fixes

- Check existing issues first
- Create an issue if one doesn't exist
- Include steps to reproduce
- Add tests that would have caught the bug

### New Features

- Discuss major features in an issue first
- Ensure it aligns with project goals (see Doctrine of Intent)
- Add comprehensive tests
- Update documentation

### CLI Commands

When adding new CLI commands:

1. Determine which module the command belongs to (`cli/core.py`, `cli/mp02.py`, etc.)
2. Follow the existing command pattern using argparse
3. Add comprehensive help text
4. Add integration tests in `tests/test_cli_integration.py`
5. Document in `docs/guide/cli.md`

### LLM Providers

When adding new LLM providers:

1. Create module in `src/intentlog/llm/`
2. Implement the abstract `LLMProvider` interface
3. Register in `llm/registry.py`
4. Make dependencies optional
5. Add tests in `tests/test_llm.py`

### MP-02 Protocol Extensions

When extending the MP-02 protocol:

1. Review `mp-02-spec.md` for normative requirements
2. Implement in appropriate `src/intentlog/mp02/` module
3. Ensure backward compatibility
4. Add comprehensive tests
5. Update `docs/guide/mp02.md`

### Integrations

When adding integrations (like Memory Vault):

1. Create module in `src/intentlog/integrations/`
2. Make it optional (handle ImportError gracefully)
3. Add tests in `tests/test_integrations.py`
4. Document in `INTEGRATION.md`
5. Add example usage in `examples/`

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `src/intentlog/__init__.py`
2. Update CHANGELOG.md
3. Run full test suite across Python versions
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. Create GitHub release
7. Publish to PyPI (when ready)

## Getting Help

- Check existing documentation in `docs/`
- Search existing issues
- Create a new issue with your question
- Be specific about your environment and problem

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intent

## License

By contributing, you agree that your contributions will be licensed under the CC BY-SA 4.0 license.

## Questions?

Feel free to open an issue for any questions about contributing!

---

Thank you for contributing to IntentLog!

# Contributing to IntentLog

Thank you for your interest in contributing to IntentLog! This document provides guidelines and instructions for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of version control concepts

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kase1111-hash/IntentLog.git
   cd IntentLog
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

3. **Run tests to verify setup**:
   ```bash
   python -m pytest tests/ -v
   ```

## Project Structure

```
IntentLog/
├── .github/
│   └── workflows/          # GitHub Actions workflows
├── src/
│   └── intentlog/          # Main package
│       ├── core.py         # Core functionality
│       ├── audit.py        # Auditing tools
│       ├── cli.py          # Command-line interface
│       └── integrations/   # External integrations
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── examples/               # Usage examples
├── docs/                   # Documentation (markdown files)
└── code/                   # Reference implementations
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
   python -m pytest tests/ -v
   ```

4. **Run audit checks**:
   ```bash
   python scripts/audit_intents.py intent_audit.log
   ```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Test both success and failure cases

Example test structure:
```python
def test_feature_does_something():
    """Test that feature does what it's supposed to do"""
    # Arrange
    setup_data = ...

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result == expected_value
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all new code
- Update INTEGRATION.md for integration changes
- Create examples in `examples/` for major features

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
   - Ensure all tests pass

## Types of Contributions

### Bug Fixes

- Check existing issues first
- Create an issue if one doesn't exist
- Include steps to reproduce
- Add tests that would have caught the bug

### New Features

- Discuss major features in an issue first
- Ensure it aligns with project goals
- Add comprehensive tests
- Update documentation

### Documentation

- Fix typos and clarify language
- Add examples and use cases
- Improve API documentation
- Update integration guides

### Performance Improvements

- Benchmark before and after
- Document the improvement
- Ensure tests still pass

## Integration Development

When adding integrations (like Memory Vault):

1. Create module in `src/intentlog/integrations/`
2. Make it optional (handle ImportError gracefully)
3. Add tests in `tests/test_integrations.py`
4. Document in `INTEGRATION.md`
5. Add example usage in `examples/`

## Release Process

(For maintainers)

1. Update version in `setup.py` and `pyproject.toml`
2. Update CHANGELOG (if exists)
3. Run full test suite
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. Create GitHub release
7. Publish to PyPI (when ready)

## Getting Help

- Check existing documentation
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

Thank you for contributing to IntentLog! 🎉

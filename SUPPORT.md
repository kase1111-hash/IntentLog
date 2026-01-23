# Support

Thank you for using IntentLog! This document provides guidance on how to get help.

## Documentation

Before seeking support, please check the available documentation:

- **[README.md](README.md)** - Project overview and quick start guide
- **[Installation Guide](docs/getting-started/installation.md)** - Detailed installation instructions
- **[Quick Start Tutorial](docs/getting-started/quickstart.md)** - Getting started with IntentLog
- **[CLI Reference](docs/guide/cli.md)** - Complete command-line interface documentation
- **[API Reference](docs/api/index.md)** - Python API documentation
- **[MP-02 Protocol](docs/guide/mp02.md)** - Protocol specification guide
- **[Integration Guide](INTEGRATION.md)** - Third-party integrations

## Getting Help

### GitHub Issues

For bug reports and feature requests, please use GitHub Issues:

- **[Report a Bug](https://github.com/kase1111-hash/IntentLog/issues/new?template=bug_report.md)** - Found something broken? Let us know!
- **[Request a Feature](https://github.com/kase1111-hash/IntentLog/issues/new?template=feature_request.md)** - Have an idea? We'd love to hear it!
- **[Browse Issues](https://github.com/kase1111-hash/IntentLog/issues)** - See existing issues and discussions

### Before Opening an Issue

1. **Search existing issues** - Your question may have already been answered
2. **Check the documentation** - The answer might be in the docs
3. **Provide context** - Include your environment, version, and steps to reproduce

### What to Include in Bug Reports

- IntentLog version (`ilog --version`)
- Python version (`python --version`)
- Operating system and version
- Complete error messages and stack traces
- Steps to reproduce the issue
- Expected vs. actual behavior

## Common Issues

### Installation Problems

```bash
# Ensure you have Python 3.8+
python --version

# Install with all dependencies
pip install intentlog[all]

# Or install specific features
pip install intentlog[crypto]    # Cryptographic features
pip install intentlog[openai]    # OpenAI integration
pip install intentlog[anthropic] # Anthropic integration
```

### Permission Issues

```bash
# Ensure proper permissions for key files
chmod 600 ~/.intentlog/*.key
```

### LLM Provider Configuration

```bash
# Set environment variables for LLM providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use local models with Ollama
ollama pull llama2
ilog config --llm-provider ollama
```

## Security Issues

**Do not report security vulnerabilities through GitHub Issues.**

Please see our [Security Policy](SECURITY.md) for information on reporting security vulnerabilities responsibly.

## Contributing

Interested in contributing? See our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup instructions
- Code style guidelines
- Testing requirements
- Pull request process

## Project Status

IntentLog is currently in **alpha** (v0.1.0). While functional, the API may change before the 1.0 release. We welcome feedback on the current design and implementation.

### Version Support

| Version | Status | Support Level |
|---------|--------|---------------|
| 0.1.x   | Alpha  | Active development, bug fixes |
| < 0.1   | Legacy | No support |

## Community

- **GitHub Repository**: [kase1111-hash/IntentLog](https://github.com/kase1111-hash/IntentLog)
- **Issues & Discussions**: [GitHub Issues](https://github.com/kase1111-hash/IntentLog/issues)

## License

IntentLog is licensed under [CC BY-SA 4.0](LICENSE.md). See the license file for details.

---

*If you find IntentLog useful, consider starring the repository on GitHub!*

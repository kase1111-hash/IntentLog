# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Basic Installation

```bash
pip install intentlog
```

## Installation with Optional Dependencies

IntentLog has zero required dependencies by design. Optional features require additional packages:

### Cryptographic Features

For Ed25519 signing and Fernet encryption:

```bash
pip install intentlog[crypto]
```

### LLM Integration

For OpenAI-powered semantic features:

```bash
pip install intentlog[openai]
```

For Anthropic Claude:

```bash
pip install intentlog[anthropic]
```

### All Features

Install everything:

```bash
pip install intentlog[all]
```

## Development Installation

For contributing to IntentLog:

```bash
git clone https://github.com/kase1111-hash/IntentLog.git
cd IntentLog
pip install -e ".[dev,all]"
```

## Verification

Verify installation:

```bash
ilog --version
# intentlog 0.1.0
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [CLI Reference](../guide/cli.md)
- [API Reference](../api/index.md)

# Changelog

All notable changes to IntentLog will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-10

**"The One That Works"** — A focused refactor that cuts feature creep, adds git integration, and ships a working CLI.

### Added

#### Git Integration (Phase 3)
- `src/intentlog/git.py` — New module for git detection, context capture, and hook management
- `ilog init` now auto-detects git repos and stores `git_root` in config
- `ilog commit` captures git branch, HEAD commit hash, and staged files as `metadata.git_context`
- `ilog log --git-commit <hash>` filters intents by associated git commit
- `ilog log --json` and `ilog status --json` for scripting
- `ilog show <id>` displays a single intent with full metadata and git context
- `ilog blame <file>` shows intent reasoning alongside git log entries
- `ilog hooks install|uninstall|status` manages a `prepare-commit-msg` git hook
- `ProjectConfig.git_root` field for persistent git repo association

#### Developer Experience
- `--json` flag on `log`, `status`, and `show` commands for machine-readable output
- Pytest markers for selective test execution: `analytics`, `context`, `llm`, `load`
- 46 new git integration tests (`tests/test_git_integration.py`)

### Changed
- Restructured `__init__.py` to lazy-load non-core modules via `__getattr__` — package now loads without `cryptography` installed
- Fixed `except ImportError` → `except BaseException` for pyo3/Rust binding panics in crypto.py and test_phase2.py
- Atomic file writes in storage.py (write-to-temp + `os.replace()`) for crash safety
- CLI uses dynamic `importlib.import_module` registration for optional command groups
- Version bumped to 0.2.0 across pyproject.toml, `__init__.py`, cli, and storage

### Removed

#### Deleted Modules (~8,600 LOC)
- `src/intentlog/integrations/` (5 files) — Boundary Daemon, SIEM, LLM classifier, Memory Vault integrations (inverted dependency direction)
- `src/intentlog/mp02/` (7 files) — MP-02 protocol (extracted to separate repo)
- `src/intentlog/triggers.py` — Workflow orchestration (premature)
- `src/intentlog/sufficiency.py` — Quality scoring (premature)
- `src/intentlog/ratelimit.py` — Rate limiting (no users hitting limits)
- `src/intentlog/privacy.py` — Full encryption/ACL system (premature)
- `src/intentlog/cli/mp02.py`, `cli/privacy.py` — CLI for deleted modules
- `tests/test_integrations.py`, `test_llm_classifier.py`, `test_mp02.py`, `test_triggers.py`, `test_privacy.py`

#### Deleted Documentation
- `AUDIT_REPORT.md`, `SECURITY_AUDIT.md` — Premature for alpha
- `Memory-Vault-Integration.md`, `INTEGRATION.md` — References deleted code
- `Your-Work-Isnt-Worthless.md` — Motivational content, not software docs
- `mp-02-spec.md`, `docs/api/mp02.md`, `docs/api/privacy.md`, `docs/guide/integrations.md`, `docs/guide/mp02.md`

### Fixed
- `pyo3_runtime.PanicException` escaping `try/except` blocks — Python's `cryptography` library raises `BaseException` subclasses, not `Exception`
- `validate_intent_name` import that referenced a non-existent function

---

## [0.1.0-alpha] - 2026-01-02

First alpha release with comprehensive production readiness improvements.

### Added

#### Production Infrastructure
- Backup and recovery module (`src/intentlog/backup.py`)
- Load testing suite (`tests/test_load.py`)

#### Security & Validation
- Input validation module (`src/intentlog/validation.py`)
- JSON schema validation (`src/intentlog/schema.py`)

#### Developer Experience
- CLI shell completion (bash, zsh, fish)
- Structured logging module
- Cross-platform file locking

#### PyPI Publication
- Updated pyproject.toml with URLs and keywords
- Optional dependency groups (`crypto`, `openai`, `anthropic`, `llm`, `all`, `docs`)

### Changed
- Refactored CLI from monolithic cli.py into modular cli/ package

## [0.1.0] - 2025-01-01

### Added
- Core CLI: `ilog init`, `commit`, `log`, `search`, `branch`, `status`
- Merkle tree chain linking with SHA-256
- Ed25519 signature support
- MP-02 Protocol implementation
- Analytics, metrics, and export
- Context management and `@intent_logger` decorator
- Privacy controls with Fernet encryption
- Deferred formalization with LLM
- Human-in-the-loop triggers
- Pluggable LLM providers (OpenAI, Anthropic, Ollama)

## [0.0.1] - Initial Development

### Added
- Initial project structure
- Core data models (Intent, IntentLog)
- Basic documentation (README, Doctrine of Intent)

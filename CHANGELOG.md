# Changelog

All notable changes to IntentLog will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Concurrent writes no longer lose intents. Writers hold one exclusive lock across each read-modify-write, and `FileLock` is re-entrant per thread. Lock files live in `.intentlog/temp/locks/`, which `ilog init` has always gitignored. `IntentLogStorage.lock_branch()` exposes the lock to callers.
- `ilog tag`, `ilog link`, `ilog merge`, `IntentLogStorage.add_intent()`, `create_branch()` and `migrate_to_chain()` no longer strip chain hashes and Ed25519 signatures from hash-chained branches. Unchanged intents are kept verbatim, and only intents from the first changed one onward are re-chained.
- `BackupManager.create_backup()` and `restore_backup()` no longer crash with a `TypeError`. `log_context()` now stores unknown fields in `extra`.
- Backups no longer include private signing keys unless `include_keys=True`. Keys saved as `keys/<name>.key` were previously archived. Restoring over an existing project keeps that project's private keys when they match the restored public keys; any other key stays in the moved-aside copy, and the restore warns about every public key left without its private key.
- Restoring a backup only ever writes regular files and directories under `.intentlog/`. The whole archive is checked, including for duplicate or aliased members and backslashes in names, before anything is written or an existing project is moved aside. A failed restore puts the previous project back, and earlier `.intentlog.bak` copies are never deleted (later ones are named `.intentlog.bak2`, and so on).
- `check_project_health()` reports real chain verification errors instead of always failing on a missing attribute.
- The git hooks no longer change commit messages. `prepare-commit-msg` shows the latest intent as a marked comment block, only for editor commits, and a new `commit-msg` hook removes that block before git stores the message. Intent text cannot end the block early. Hooks installed by earlier versions show as `outdated` in `ilog hooks status`; re-run `ilog hooks install` to update them.
- `ilog commit` prints the intent hash that `log`, `show`, `search`, `tag` and `link` use. Lookups also accept a chain-hash prefix.
- `ilog link` finds targets on other branches.
- Invalid `--branch` names in `tag` and `link` print an error instead of a traceback, and so does a branch lock that stays held for more than 10 seconds.
- `ilog link` no longer holds the source branch's lock while reading other branches, which could deadlock two links in opposite directions.
- `ilog tag` no longer rewrites an intent when no tag changed.
- `ilog chain verify` reports entries whose chain data was removed or cannot be read, including the first entry, instead of crashing or re-hashing the branch as if it had never been chained. `ilog commit` and `IntentLogStorage.add_intent()` refuse to append to such a branch, and `ilog chain migrate --force` rebuilds the chain from the first damaged entry, even when no entry is readable.
- Branch names are validated on every storage path, so no branch name or `config.json` value can address a file outside `.intentlog/`.
- Entries that cannot be read are never dropped silently. `ilog commit`, `ilog chain migrate`, `tag`, `link`, `merge` and `add_intent()` refuse to rewrite a branch that would lose them, on chained and legacy branches alike; `ilog chain migrate --force --branch <name>` drops them explicitly. `ilog branch` copies the source verbatim, and the `@intent_logger` decorator logs a warning when it cannot save an intent instead of failing silently.
- `ilog merge` checks the source branch as well as the target, so a tampered or unreadable source is not re-chained into a clean-looking target without `--force`.
- A chained branch whose `chain_version` or `chain_hash` markers were removed is still treated as a damaged chain, so its signatures are never stripped by re-chaining it as legacy data.
- Date-and-time values with a `Z` UTC suffix are read on Python 3.8-3.10 as they are on 3.11+, in chained and legacy entries alike, so a chain that verifies on one supported Python verifies on all of them.
- `mkdocs build --strict` passes again, and the API reference only lists members that exist. The `docs` extra allows mkdocstrings 1.x, which works with current pymdown-extensions.
- `examples/basic_usage.py`, `benchmarks/run_benchmarks.py` and the pytest-benchmark suite run against the v0.2.0 API.
- `MANIFEST.in` no longer lists files that were deleted, and `setup.py` defers all metadata to `pyproject.toml`.

### Added
- `--force` for `ilog tag` and `ilog link`. Without it they refuse a change that would remove signatures, or that would rewrite chain entries which already fail verification.
- `--force` for `ilog merge` and `ilog chain migrate`, which otherwise refuse to rewrite a damaged stored chain. `migrate_to_chain()` and `add_intent()` take `force=`.
- `IntentLogStorage.save_intents()` takes `strict=` and returns the number of signatures removed. New exceptions: `SignatureLossError` and `ChainIntegrityError`.

### Changed
- The ruff lint job is clean across `src/` and `tests/`.

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

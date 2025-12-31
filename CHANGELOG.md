# Changelog

All notable changes to IntentLog will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Python 3.12 support in classifiers
- Optional dependency groups in pyproject.toml (`crypto`, `openai`, `anthropic`, `all`)
- Coverage reporting in GitHub Actions CI
- Multi-version Python testing matrix (3.9, 3.10, 3.11, 3.12)
- This CHANGELOG file

### Changed
- Updated GitHub Actions from setup-python@v4 to setup-python@v5

## [0.1.0] - 2025-01-01

### Added

#### Phase 1: Core CLI
- `ilog init` - Initialize IntentLog repository
- `ilog commit` - Record intent with prose description
- `ilog log` - View intent history
- `ilog search` - Search intents by content
- `ilog branch` - Branch management
- `ilog status` - Show project status
- Persistent storage with JSON serialization

#### Phase 2: Cryptographic Integrity
- Merkle tree chain linking with SHA-256
- Ed25519 signature support
- `ilog keys generate` - Generate signing keypairs
- `ilog keys sign` - Sign intents
- `ilog chain verify` - Verify chain integrity
- Inclusion proofs for individual intents

#### Phase 3: MP-02 Protocol
- Observer system for raw signal capture
- Temporal segmentation based on time windows
- Receipt generation with LLM validation
- Append-only ledger for tamper-evidence
- `ilog observe` - Start observation session
- `ilog segment` - Create time-based segments
- `ilog receipt` - Generate effort receipts
- `ilog ledger` - Manage receipt ledger

#### Phase 4: Analytics & Metrics
- Intent analytics: latency, frequency, trends, bottlenecks
- Doctrine metrics: Intent Density, Information Density, Auditability
- `ilog analytics` - View intent analytics
- `ilog metrics` - Compute doctrine metrics
- `ilog export` - Export in JSON, JSONL, CSV, HuggingFace, OpenAI formats
- `ilog sufficiency` - Test intent quality

#### Phase 5: Context & Decorator
- `@intent_logger` decorator for automatic logging
- Intent context propagation across function calls
- Session management for grouping related intents
- Context hooks (on_enter, on_exit)
- Environment variable propagation for subprocesses
- Full tracing with trace IDs and span IDs

#### Phase 6: Privacy Controls (MP-02 Section 12)
- Fernet symmetric encryption for sensitive content
- Privacy levels: PUBLIC, INTERNAL, CONFIDENTIAL, SECRET, TOP_SECRET
- Access control policies with granular permissions
- Revocation mechanism for future observation blocking
- `ilog privacy status` - View privacy status
- `ilog privacy revoke` - Revoke observation consent
- `ilog privacy encrypt` - Encrypt sensitive intents
- `ilog privacy keys` - Manage encryption keys

#### Phase 8: Deferred Formalization
- Derive code, rules, heuristics, schemas from prose intent
- LLM-powered formalization with provenance tracking
- Multiple output types: code, rules, heuristics, schema, config, spec, tests
- `ilog formalize` - Formalize intents into structured outputs

#### Phase 9: Human-in-the-Loop Triggers
- Trigger types: notification, confirmation, approval, review
- Sensitivity levels: low, medium, high, critical
- Console and callback-based handlers
- Timeout and escalation support
- Audit trail integration

#### LLM Integration
- Pluggable LLM provider architecture
- OpenAI provider (GPT-4, embeddings)
- Anthropic Claude provider (completions)
- Ollama provider (local models)
- Mock provider for testing
- Semantic diff between branches
- Semantic search with embeddings

#### Security
- Zero production dependencies
- Optional cryptography for signing/encryption
- Input validation and sanitization
- Secure key management

### Fixed
- Low-severity security issues identified in security audit

## [0.0.1] - Initial Development

### Added
- Initial project structure
- Core data models (Intent, IntentLog)
- Basic documentation (README, Doctrine of Intent)
- MP-02 protocol specification

# MP-02 — Proof-of-Effort Receipt Protocol

*NatLangChain Effort Verification Specification (Standalone)*

**Status**: Draft (Normative)
**Last Updated**: December 2025

---

## 1. Purpose

MP-02 defines the protocol by which human intellectual effort is observed, validated, and recorded as cryptographically verifiable receipts on NatLangChain.

The protocol establishes a primitive that is:

- Verifiable without trusting the issuer
- Human-readable over long time horizons
- Composable with negotiation, licensing, and settlement protocols

MP-02 does not assert value, ownership, or compensation. It asserts that effort occurred, with traceable provenance.

## 2. Design Principles

- **Process Over Artifact** — Effort is validated as a process unfolding over time, not a single output.
- **Continuity Matters** — Temporal progression is a primary signal of genuine work.
- **Receipts, Not Claims** — The protocol records evidence, not conclusions about value.
- **Model Skepticism** — LLM assessments are advisory and must be reproducible.
- **Partial Observability** — Uncertainty is preserved, not collapsed.

## 3. Definitions

### 3.1 Effort

A temporally continuous sequence of human cognitive activity directed toward an intelligible goal.

### 3.2 Signal

A raw observable trace of effort, including but not limited to:

- Voice transcripts
- Text edits
- Command history
- Structured tool interaction

### 3.3 Effort Segment

A bounded time slice of signals treated as a unit of analysis.

### 3.4 Receipt

A cryptographic record attesting that a specific effort segment occurred, with references to its source signals and validation metadata.

## 4. Actors

### 4.1 Human Worker

The individual whose effort is being recorded.

### 4.2 Observer

A system component responsible for capturing raw signals.

### 4.3 Validator

An LLM-assisted process that analyzes effort segments for coherence and progression.

### 4.4 Ledger

An append-only system that anchors receipts and their hashes.

## 5. Effort Capture

Observers MAY record:

- Continuous or intermittent signals
- Multi-modal inputs

Observers MUST:

- Time-stamp all signals
- Preserve ordering
- Disclose capture modality

Observers MUST NOT:

- Alter raw signals
- Infer intent beyond observed data

## 6. Segmentation

Signals are grouped into Effort Segments based on:

- Time windows
- Activity boundaries
- Explicit human markers

Segmentation rules MUST be deterministic and disclosed.

## 7. Validation

Validators MAY assess:

- Linguistic coherence
- Conceptual progression
- Internal consistency
- Indicators of synthesis vs duplication

Validators MUST:

- Produce deterministic summaries
- Disclose model identity and version
- Preserve dissent and uncertainty

Validators MUST NOT:

- Declare effort as valuable
- Assert originality or ownership
- Collapse ambiguous signals into certainty

## 8. Receipt Construction

Each Effort Receipt MUST include:

- Receipt ID
- Time bounds
- Hashes of referenced signals
- Deterministic effort summary
- Validation metadata
- Observer and Validator identifiers

Receipts MAY reference:

- Prior receipts
- External artifacts

## 9. Anchoring

Receipts are anchored by:

1. Hashing receipt contents
2. Appending hashes to a ledger

The ledger MUST be:

- Append-only
- Time-ordered
- Publicly verifiable

## 10. Verification

A third party MUST be able to:

- Recompute receipt hashes
- Inspect validation metadata
- Confirm ledger inclusion

Trust in the Observer or Validator is not required.

## 11. Failure Modes

The protocol explicitly records:

- Gaps in observation
- Conflicting validations
- Suspected manipulation
- Incomplete segments

Failures reduce confidence but do not invalidate receipts.

## 12. Privacy and Agency

- Raw signals MAY be encrypted or access-controlled
- Receipts MUST NOT expose raw content by default
- Humans MAY revoke future observation
- Past receipts remain immutable

## 13. Non-Goals

MP-02 does NOT:

- Measure productivity
- Enforce labor conditions
- Replace authorship law
- Rank humans by output

## 14. Compatibility

MP-02 receipts are compatible with:

- MP-01 Negotiation & Ratification
- Licensing and delegation modules
- External audit systems

## 15. Canonical Rule

> If effort cannot be independently verified as having occurred over time, it must not be capitalized.

---

# Implementation Status & Roadmap

This section tracks the implementation status of all features documented across IntentLog.

## Legend

| Status | Meaning |
|--------|---------|
| Implemented | Feature is fully working in code |
| Partial | Code structure exists but functionality incomplete |
| Stub | CLI/API defined but not implemented |
| Planned | Documented but no code exists |
| Future | Listed for future consideration |

---

## Core Features Status

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Intent data structure | Implemented | `src/intentlog/core.py:14` | Intent class with ID, name, reasoning, timestamp, metadata, parent |
| IntentLog manager | Implemented | `src/intentlog/core.py:45` | Add, get, search, export, chain traversal |
| Parent-child intent chains | Implemented | `src/intentlog/core.py:69` | `get_intent_chain()` method |
| Intent validation | Implemented | `src/intentlog/core.py:36` | Requires name and non-empty reasoning |
| Basic text search | Implemented | `src/intentlog/core.py:90` | Case-insensitive search in name/reasoning |
| Export to dict/JSON | Implemented | `src/intentlog/core.py:102` | `export_to_dict()` method |
| Audit: empty reasoning | Implemented | `src/intentlog/audit.py:40` | Detects empty `intent_reasoning` |
| Audit: loop detection | Implemented | `src/intentlog/audit.py:46` | Configurable repeat threshold |
| Memory Vault integration | Implemented | `src/intentlog/integrations/memory_vault.py` | Classification, store, recall |
| Intent classification | Implemented | `src/intentlog/integrations/memory_vault.py:125` | Keyword-based, 5 levels (0-5) |
| **Storage module** | **Implemented** | `src/intentlog/storage.py` | Persistent `.intentlog/` storage, JSON serialization, file locking |
| **Project configuration** | **Implemented** | `src/intentlog/storage.py:25` | ProjectConfig with branch tracking |
| **Intent hash computation** | **Implemented** | `src/intentlog/storage.py:74` | SHA-256 canonical JSON hashing |
| **Branch management** | **Implemented** | `src/intentlog/storage.py:285` | Create, switch, list branches |

## CLI Commands Status

| Command | Status | Location | Notes |
|---------|--------|----------|-------|
| `ilog init <project>` | **Implemented** | `cli.py:24` | Creates `.intentlog/` with config, intents, branches |
| `ilog commit <message>` | **Implemented** | `cli.py:43` | Adds intent with hash, persists to JSON |
| `ilog branch <name>` | **Implemented** | `cli.py:92` | Creates/switches branches, copies intents |
| `ilog log` | **Implemented** | `cli.py:131` | Shows history with `--limit` and `--branch` |
| `ilog search <query>` | **Implemented** | `cli.py:186` | Case-insensitive search in name/reasoning |
| `ilog audit <file>` | **Implemented** | `cli.py:235` | Full functionality with `audit.py` |
| `ilog status` | **Implemented** | `cli.py:248` | Shows project info, branch, intent count |
| `ilog diff` | **Implemented** | `cli.py:316` | Semantic diff between branches (with LLM) |
| `ilog merge` | **Implemented** | `cli.py:392` | Merge branches with optional message |
| `ilog config` | **Implemented** | `cli.py:441` | Configure LLM provider and settings |
| `--attach` flag | **Implemented** | `cli.py:59` | Attaches git files with hashes to metadata |
| `--semantic` flag | **Implemented** | `cli.py:202` | Semantic search using embeddings |

---

## Unimplemented Features by Category

### Category A: Core Version Control (Priority: Critical)

Features required for basic IntentLog functionality.

| Feature | Source | Description | Status |
|---------|--------|-------------|--------|
| Persistent storage | INTEGRATION.md | Save/load intent logs to `.intentlog/` directory | **Implemented** |
| Prose commits with hashes | README.md | Timestamped commits with SHA-256 hashes | **Implemented** |
| Merkle tree integrity | README.md | Hash-chain for tamper-evident history | Partial (hashes per intent, no chain yet) |
| Intent branching | README.md | Create experimental branches for alternatives | **Implemented** |
| Merge via explanation | README.md | Resolve conflicts with narrative commits | **Implemented** |
| Precedent trails | README.md | Reference chains between commits (case law) | Partial (parent_id supported) |
| File attachment | README.md | `--attach` to link code/files to commits | **Implemented** |

### Category B: LLM-Powered Features (Priority: High, Target: Q1 2026)

Features requiring LLM integration.

| Feature | Source | Description | Status |
|---------|--------|-------------|--------|
| Semantic diffs | README.md | LLM-generated human-readable change summaries | **Implemented** |
| Semantic search | README.md | Query reasoning with natural language | **Implemented** |
| Deferred formalization | README.md | LLM derives code/rules from prose on demand | Planned |
| Automated classification | INTEGRATION.md | LLM-based intent classification | Partial (keyword-based) |
| Conflict resolution via LLM | INTEGRATION.md | LLM-assisted merge reasoning | **Implemented** |
| Pluggable LLM backends | README.md | Support OpenAI, Anthropic, local models | **Implemented** |

### Category C: MP-02 Protocol Components (Priority: Medium)

Components defined in MP-02 specification sections 4-12.

| Feature | Spec Section | Description | Status |
|---------|--------------|-------------|--------|
| Observer system | Section 5 | Capture raw signals with timestamps | Planned |
| Validator (LLM-assisted) | Section 7 | Assess coherence and progression | Planned |
| Effort Segmentation | Section 6 | Group signals into bounded segments | Planned |
| Receipt construction | Section 8 | Build cryptographic receipts | Planned |
| Ledger anchoring | Section 9 | Append-only, time-ordered log | Planned |
| Third-party verification | Section 10 | Recompute hashes, verify inclusion | Planned |
| Failure mode recording | Section 11 | Track gaps, conflicts, manipulation | Planned |
| Privacy controls | Section 12 | Encryption, access control, revocation | Planned |

### Category D: Advanced Use Cases (Priority: Medium)

Features from Advanced-Use-Cases.md for production deployments.

| Feature | Description | Status |
|---------|-------------|--------|
| `@intent_logger` decorator | Automatic nested intent tracing for functions | Planned |
| Eval set generation | Export intents as ground truth for evaluation | Planned |
| Latency tracking | Timestamp start/end for bottleneck discovery | Planned |
| Human-in-the-loop triggers | Show intent before sensitive operations | Planned |
| Fine-tuning data pipeline | Filter logs for model training data | Planned |
| `session_id` context | Trace user journeys across sessions | Planned |
| Conditional logging levels | Granular vs high-level by environment | Planned |

### Category E: Doctrine of Intent Metrics (Priority: Future)

Metrics from Doctrine-of-intent.md for provenance verification.

| Feature | Description | Status |
|---------|-------------|--------|
| Intent Density (Di) scoring | Measure resolution and continuity of records | Future |
| Intent Sufficiency Test | Validate continuity, directionality, resolution, anchoring, attribution | Future |
| Information Density metrics | Measure auditability and fraud resistance | Future |

### Category F: Infrastructure & Integrations (Priority: Future)

Long-term infrastructure improvements from INTEGRATION.md.

| Feature | Description | Status |
|---------|-------------|--------|
| Decentralized storage | Optional distributed storage backends | Future |
| Hardware (TPM) binding | Bind critical intents to hardware security | Future |
| Multi-language support | Beyond Python (JS, Go, Rust) | Future |
| IDE plugins | Inline intent logging in VSCode, JetBrains | Future |
| Git deep integration | Bidirectional sync with git history | Future |
| External anchoring | Bitcoin/Ethereum timestamping | Future |

---

# Implementation Plans

## Plan 1: Core CLI Implementation

**Priority**: Critical
**Prerequisite for**: All other plans
**Estimated Scope**: ~500 lines of code

### Goal
Make all CLI commands functional with persistent storage.

### Implementation Steps

1. **Create storage module** (`src/intentlog/storage.py`)
   - Define `.intentlog/` directory structure
   - JSON schema for intents: `{ intents: [...], branches: {...}, config: {...} }`
   - Methods: `init_project()`, `save_intents()`, `load_intents()`
   - Handle file locking for concurrent access

2. **Implement `ilog init`** (`cli.py:14`)
   - Create `.intentlog/` directory
   - Initialize `config.json` with project name, created timestamp
   - Create empty `intents.json`
   - Create `.intentlog/.gitignore` for sensitive data

3. **Implement `ilog commit`** (`cli.py:22`)
   - Load existing intents
   - Create Intent with auto-generated UUID and timestamp
   - Compute SHA-256 hash of content
   - Append to intents list
   - Save to disk
   - Print commit hash

4. **Implement `ilog branch`** (`cli.py:30`)
   - Add `branches` section to config
   - Track current branch in config
   - Create branch-specific intent files
   - Copy intents at branch point

5. **Implement `ilog log`** (`cli.py:38`)
   - Load intents from current branch
   - Format: `[hash] timestamp | intent_name: reasoning`
   - Support `--limit N` for pagination
   - Support `--branch name` to view other branches

6. **Implement `ilog search`** (`cli.py:45`)
   - Load intents
   - Call `IntentLog.search_intents(query)`
   - Display formatted results with context

7. **Implement `--attach`** (`cli.py:88`)
   - Run `git ls-files --cached`
   - Store file list in intent metadata
   - Optionally store file hashes

### Files to Create/Modify
- `src/intentlog/storage.py` (new, ~150 lines)
- `src/intentlog/cli.py` (modify, ~100 lines additional)
- `src/intentlog/core.py` (modify, ~30 lines for hash)

### Testing
- `tests/test_storage.py` (new)
- `tests/test_cli_integration.py` (new)

---

## Plan 2: Cryptographic Integrity

**Priority**: High
**Depends on**: Plan 1
**Estimated Scope**: ~400 lines of code

### Goal
Implement Merkle tree and cryptographic signatures for tamper-evident history.

### Implementation Steps

1. **Add hash computation** (`src/intentlog/crypto.py`)
   - `hash_intent(intent) -> str`: SHA-256 of canonical JSON
   - Canonical JSON: sorted keys, no whitespace
   - Include: intent_id, name, reasoning, timestamp, parent_id, metadata

2. **Implement Merkle tree** (`src/intentlog/merkle.py`)
   - Each intent stores `prev_hash` (hash of previous intent)
   - `compute_root_hash(intents) -> str`
   - `verify_chain(intents) -> (bool, errors)`
   - Handle branch divergence

3. **Add cryptographic signatures** (`src/intentlog/crypto.py`)
   - Support Ed25519 (PyNaCl) or GPG
   - `sign_intent(intent, key) -> signature`
   - `verify_signature(intent, signature, pubkey) -> bool`
   - Store signature in intent metadata

4. **Create verification CLI** (`cli.py`)
   - `ilog verify`: Check integrity of all intents
   - `ilog verify --deep`: Re-verify all signatures
   - Report: missing hashes, broken chains, invalid signatures

5. **Key management**
   - `ilog keys generate`: Create new keypair
   - `ilog keys export`: Export public key
   - Store keys in `.intentlog/keys/`

### Files to Create/Modify
- `src/intentlog/crypto.py` (new, ~150 lines)
- `src/intentlog/merkle.py` (new, ~100 lines)
- `src/intentlog/cli.py` (add verify commands, ~50 lines)
- `src/intentlog/core.py` (add hash field, ~20 lines)

---

## Plan 3: LLM Integration Layer

**Priority**: Medium
**Depends on**: Plan 1
**Target**: Q1 2026
**Estimated Scope**: ~800 lines of code

### Goal
Add pluggable LLM backend for semantic features.

### Implementation Steps

1. **Create provider interface** (`src/intentlog/llm/provider.py`)
   ```python
   class LLMProvider(ABC):
       @abstractmethod
       def complete(self, prompt: str) -> str: ...
       @abstractmethod
       def embed(self, text: str) -> List[float]: ...
   ```

2. **Implement providers**
   - `openai.py`: OpenAI API integration
   - `anthropic.py`: Claude API integration
   - `local.py`: Local model support (ollama, llama.cpp)

3. **Implement semantic diff** (`src/intentlog/semantic.py`)
   - `semantic_diff(intent_a, intent_b) -> str`
   - Generate prompt: "Compare these two intents and describe the changes..."
   - Cache results keyed by intent hashes

4. **Implement semantic search** (`src/intentlog/semantic.py`)
   - `embed_intent(intent) -> List[float]`
   - Store embeddings alongside intents
   - `semantic_search(query, intents) -> List[Intent]`
   - Use cosine similarity

5. **Configuration**
   - Add to `.intentlog/config.json`:
     ```json
     {
       "llm": {
         "provider": "openai",
         "model": "gpt-4",
         "api_key_env": "OPENAI_API_KEY"
       }
     }
     ```

6. **CLI integration**
   - `ilog diff branch1..branch2`: Semantic diff
   - `ilog search --semantic "query"`: Semantic search

### Files to Create
- `src/intentlog/llm/__init__.py`
- `src/intentlog/llm/provider.py` (~50 lines)
- `src/intentlog/llm/openai.py` (~100 lines)
- `src/intentlog/llm/anthropic.py` (~100 lines)
- `src/intentlog/semantic.py` (~200 lines)

---

## Plan 4: MP-02 Observer & Validator

**Priority**: Medium
**Depends on**: Plans 1, 2
**Estimated Scope**: ~600 lines of code

### Goal
Implement Observer and Validator components per MP-02 specification.

### Implementation Steps

1. **Create Observer base class** (`src/intentlog/mp02/observer.py`)
   ```python
   class Observer(ABC):
       @abstractmethod
       def capture(self) -> Signal: ...
       @abstractmethod
       def start(self): ...
       @abstractmethod
       def stop(self): ...
   ```

2. **Implement text observer**
   - File watcher for text edits
   - Keystroke capture (opt-in)
   - Terminal command history

3. **Create segmentation engine** (`src/intentlog/mp02/segmentation.py`)
   - Time-window segmentation (configurable duration)
   - Activity boundary detection (pause detection)
   - Explicit markers via `ilog segment start/end`

4. **Create Validator framework** (`src/intentlog/mp02/validator.py`)
   - Coherence checking via LLM
   - Progression analysis
   - Deterministic summary generation
   - Model version tracking

5. **Implement Receipt builder** (`src/intentlog/mp02/receipt.py`)
   - Construct receipt with all required fields (Section 8)
   - Hash and sign receipt
   - Reference prior receipts
   - Attach external artifacts

### Files to Create
- `src/intentlog/mp02/__init__.py`
- `src/intentlog/mp02/observer.py` (~150 lines)
- `src/intentlog/mp02/segmentation.py` (~100 lines)
- `src/intentlog/mp02/validator.py` (~150 lines)
- `src/intentlog/mp02/receipt.py` (~100 lines)

---

## Plan 5: Ledger & Anchoring System

**Priority**: Medium
**Depends on**: Plans 1, 2, 4
**Estimated Scope**: ~400 lines of code

### Goal
Implement append-only ledger with public verifiability per MP-02 Section 9.

### Implementation Steps

1. **Create local ledger** (`src/intentlog/ledger.py`)
   - Append-only file: `.intentlog/ledger.log`
   - Format: `timestamp|receipt_hash|prev_hash`
   - File locking for concurrent writes
   - Auto-rotation by size

2. **Implement anchoring** (`src/intentlog/anchoring.py`)
   - Periodic checkpoints (configurable interval)
   - `anchor_to_file(ledger, path)`: Export checkpoint
   - Optional: OpenTimestamps integration
   - Optional: Blockchain anchoring (Bitcoin OP_RETURN)

3. **Add verification API**
   - `verify_ledger() -> (bool, errors)`
   - `generate_inclusion_proof(receipt_id) -> Proof`
   - Third-party verification without full ledger

4. **CLI commands**
   - `ilog ledger show`: Display ledger entries
   - `ilog ledger verify`: Check integrity
   - `ilog ledger anchor`: Create external anchor
   - `ilog ledger export`: Export for verification

### Files to Create
- `src/intentlog/ledger.py` (~200 lines)
- `src/intentlog/anchoring.py` (~100 lines)

---

## Plan 6: @intent_logger Decorator

**Priority**: Low
**Depends on**: Plan 1
**Estimated Scope**: ~300 lines of code

### Goal
Automatic intent tracing for Python functions per Advanced-Use-Cases.md.

### Implementation Steps

1. **Create decorator** (`src/intentlog/decorator.py`)
   ```python
   @intent_logger(category="researcher")
   def gather_data(topic):
       # Automatically logs intent on entry/exit
       pass
   ```

2. **Add context management** (`src/intentlog/context.py`)
   - Thread-local storage for current intent
   - `get_current_intent() -> Intent`
   - Automatic parent linking for nested calls
   - Async support with contextvars

3. **Implement metadata capture**
   - Entry timestamp, exit timestamp
   - Latency calculation
   - Exception tracking
   - Return value summary

4. **Configuration options**
   - `level="high"` vs `level="granular"`
   - `session_id` injection
   - Environment-based filtering

### Files to Create
- `src/intentlog/decorator.py` (~150 lines)
- `src/intentlog/context.py` (~100 lines)

---

## Plan 7: Eval Set & Analytics

**Priority**: Low
**Depends on**: Plan 1
**Estimated Scope**: ~350 lines of code

### Goal
Export intents for evaluation and analytics per Advanced-Use-Cases.md.

### Implementation Steps

1. **Create JSON export format** (`src/intentlog/export.py`)
   - Schema: intent + expected outcome pairs
   - Filter by: latency, category, date range
   - Anonymization option

2. **Implement analytics** (`src/intentlog/analytics.py`)
   - Latency distribution statistics
   - Intent frequency by category
   - Error/correction tracking
   - Generate reports

3. **Fine-tuning data export**
   - Filter "High Latency" or "User Corrected" intents
   - Format for common frameworks (HuggingFace, OpenAI)
   - Include/exclude specific fields

### Files to Create
- `src/intentlog/export.py` (~150 lines)
- `src/intentlog/analytics.py` (~150 lines)

---

## Plan 8: Doctrine Metrics

**Priority**: Future
**Depends on**: Plans 1-4
**Estimated Scope**: ~400 lines of code

### Goal
Implement Intent Sufficiency Test per Doctrine-of-intent.md.

### Implementation Steps

1. **Intent Density scoring** (`src/intentlog/metrics.py`)
   - Resolution: decisions logged per time unit
   - Continuity: gaps between entries
   - Score: 0.0 to 1.0

2. **Intent Sufficiency Test** (`src/intentlog/sufficiency.py`)
   - Continuity check: duration spans meaningful period
   - Directionality: goals and constraints documented
   - Resolution: sufficient detail for exploration vs narration
   - Temporal anchoring: timestamps non-retroactive
   - Human attribution: responsible agent identified
   - Return: pass/fail + confidence score

3. **Information Density metrics**
   - Auditability score
   - Fraud resistance rating
   - Comparative benchmarks

### Files to Create
- `src/intentlog/metrics.py` (~200 lines)
- `src/intentlog/sufficiency.py` (~150 lines)

---

## Implementation Priority Summary

| Phase | Plans | Target | Description |
|-------|-------|--------|-------------|
| 1 | 1, 2 | Immediate | Core functionality: persistence, CLI, crypto |
| 2 | 3, 6 | Q1 2026 | LLM integration, decorator |
| 3 | 4, 5 | Q2 2026 | MP-02 protocol components |
| 4 | 7, 8 | Q3 2026 | Analytics and metrics |
| 5 | Category F | 2026+ | Infrastructure expansion |

---

## Documentation Index

This section provides a consolidated reference to all project documentation and their purposes.

### Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](README.md) | Project overview, quick start, use cases | Users, contributors |
| [mp-02-spec.md](mp-02-spec.md) | Proof-of-Effort Receipt Protocol specification + implementation tracking | Developers, architects |
| [Doctrine-of-intent.md](Doctrine-of-intent.md) | Philosophical framework: provenance-first value attribution | Strategists, researchers |
| [Prior-Art.md](Prior-Art.md) | Design patent timestamp (Dec 16, 2025), core design primitives | Legal, historians |

### Operational Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| [INTEGRATION.md](INTEGRATION.md) | Integration overview for Git, CI/CD, LLMs | DevOps, integrators |
| [Memory-Vault-Integration.md](Memory-Vault-Integration.md) | Detailed Memory Vault secure storage integration | Security engineers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, code style, PR workflow | Contributors |
| [Advanced-Use-Cases.md](Advanced-Use-Cases.md) | Production patterns: multi-agent, eval sets, HITL | Advanced users |

### Supporting Content

| Document | Purpose | Audience |
|----------|---------|----------|
| [Your-Work-Isnt-Worthless.md](Your-Work-Isnt-Worthless.md) | Practical essay on provenance value for creators | Creators, artists |
| [LICENSE.md](LICENSE.md) | CC BY-SA 4.0 full legal text | Legal |

### Cross-Reference: Features to Documentation

| Feature Area | Primary Doc | Supporting Docs |
|--------------|-------------|-----------------|
| Core vision & primitives | README.md | Prior-Art.md, Doctrine-of-intent.md |
| CLI commands | README.md (examples) | INTEGRATION.md (reference) |
| MP-02 Protocol | mp-02-spec.md (sections 1-15) | Doctrine-of-intent.md (theory) |
| Memory Vault | Memory-Vault-Integration.md | INTEGRATION.md (overview) |
| `@intent_logger` decorator | Advanced-Use-Cases.md | CONTRIBUTING.md (code style) |
| Cryptographic integrity | mp-02-spec.md (Plan 2) | Prior-Art.md (Merkle trees) |
| LLM features | mp-02-spec.md (Plan 3) | INTEGRATION.md (LLM section) |
| Doctrine metrics | mp-02-spec.md (Plan 8) | Doctrine-of-intent.md (sections 7-8) |

---

## Verification Notes

### Verified Implementation Status (December 2025)

**Phase 1 Complete** - Core CLI and storage implemented.
**Phase 2 Complete** - LLM integration with semantic features.

**Total: 105 tests passing**

---

**Phase 1 - Core CLI & Storage:**
- `Intent` dataclass with UUID, timestamp, parent linking (`core.py:14-43`)
- `IntentLog` manager with add, get, search, export, chain (`core.py:45-109`)
- Audit engine: empty reasoning + loop detection (`audit.py:23-56`)
- Memory Vault integration: classification, store, recall (`integrations/memory_vault.py`)
- **Storage module** with persistent `.intentlog/` directory (`storage.py`)
- **Project configuration** with branch tracking and LLM settings (`storage.py`)
- **Intent hash computation** using SHA-256 (`storage.py:74`)
- **Branch management** - create, switch, list (`storage.py:285`)

**Phase 2 - LLM Integration:**
- **LLM Provider Interface** - Abstract base class with MockProvider (`llm/provider.py`)
- **OpenAI Provider** - GPT-4, GPT-3.5, embeddings (`llm/openai.py`)
- **Anthropic Provider** - Claude 3.5, Claude 3 (`llm/anthropic.py`)
- **Ollama Provider** - Local models, embeddings (`llm/ollama.py`)
- **Provider Registry** - Dynamic provider registration and lookup (`llm/registry.py`)
- **Semantic Engine** - Diff, search, merge resolution (`semantic.py`)
- **Embedding Caching** - Persistent cache for embeddings

**CLI Commands (All Implemented):**
- `ilog init <project>` - Creates `.intentlog/` with config
- `ilog commit <message>` - Adds intent with hash
- `ilog branch [name]` - Lists, creates, or switches branches
- `ilog log` - Shows intent history
- `ilog search <query>` - Text or semantic search (`--semantic`)
- `ilog diff branch1..branch2` - Semantic diff between branches
- `ilog merge <branch>` - Merge with optional message
- `ilog config llm` - Configure LLM provider
- `ilog status` - Shows project info with LLM status
- `ilog audit <file>` - Audit intent logs

**Test Coverage:**
- `test_storage.py`: 27 tests for storage module
- `test_cli_integration.py`: 21 tests for CLI end-to-end
- `test_core.py`: 11 tests for core classes
- `test_audit.py`: 4 tests for audit functionality
- `test_integrations.py`: 9 tests for Memory Vault
- `test_llm.py`: 29 tests for LLM module
- **Total: 105 tests passing**

### Known Issues

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| ~~License mismatch~~ | `setup.py` | ~~Medium~~ | **Fixed** |
| ~~pytest in install_requires~~ | `setup.py` | ~~Low~~ | **Fixed** |

*No outstanding issues as of December 2025 verification.*

---

*End of MP-02 Specification and Implementation Roadmap*

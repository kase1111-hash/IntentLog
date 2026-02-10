# IntentLog Refocus Plan

**Context**: This plan follows from the [Concept-Execution Evaluation](EVALUATION.md), which classified IntentLog as **Feature Creep** — a sound core idea diluted by 3-4 separate products. This document is the concrete, ordered work plan to fix that.

**Goal**: Ship a focused, working `ilog` CLI that captures decision reasoning alongside git commits. No LLMs required. No crypto required. Just works.

**Target**: IntentLog v0.2.0 — "The One That Works"

---

## Phase 0: Fix What's Broken (Week 1)

Everything else is blocked until the test suite runs.

### 0.1 Fix the `__init__.py` import chain

**Problem**: `__init__.py` eagerly imports 15+ modules including `crypto.py`, which triggers `cryptography` Rust bindings at package load time. If anything in that chain fails, the entire package — including the CLI — is dead. Currently, 9 of 15 test modules fail to even collect.

**Fix**: Restructure `__init__.py` to lazy-import everything except `core` and `storage`. All other modules should be imported only when their functionality is explicitly requested.

```
# What __init__.py should export at load time:
from .core import Intent, IntentLog
from .storage import IntentLogStorage, ...errors...
from .merkle import MerkleChain, chain_intents, verify_chain, ...

# Everything else: lazy or removed from __init__.py entirely
```

**Modules to remove from `__init__.py` immediately** (they can still be imported directly by consumers):
- `ratelimit` — not needed at package level
- `sufficiency` — not needed at package level
- `context` — not needed at package level
- `decorator` — not needed at package level
- `triggers` — not needed at package level
- `privacy` — not needed at package level
- `backup` — not needed at package level
- `analytics` — not needed at package level
- `metrics` — not needed at package level
- `export` — not needed at package level

**Wrap crypto import properly**: The existing `try/except Exception` is correct in principle but the exception is `pyo3_runtime.PanicException` which propagates before the handler catches it. Test with a broken `cryptography` install to confirm the guard actually works.

### 0.2 Get all core tests passing

After fixing imports, run the full test suite and triage:

| Priority | Test File | Expected Status |
|----------|-----------|-----------------|
| P0 | `test_core.py` | Must pass |
| P0 | `test_storage.py` | Must pass |
| P0 | `test_phase2.py` (merkle) | Must pass |
| P0 | `test_cli_integration.py` | Must pass |
| P0 | `test_audit.py` | Must pass (already passes) |
| P1 | `test_phase4.py` (analytics) | Should pass |
| P1 | `test_phase5.py` (context) | Should pass |
| P2 | Everything else | Can be skipped/deferred |

**Acceptance criteria**: `pytest tests/test_core.py tests/test_storage.py tests/test_phase2.py tests/test_cli_integration.py tests/test_audit.py -v` — all green.

### 0.3 Add atomic file writes to storage.py

**Problem**: `storage.py` writes JSON directly to files. A crash mid-write corrupts the project. This is the #1 reliability risk.

**Fix**: Write-to-temp-then-rename pattern:
```python
def _atomic_write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic on POSIX
```

Apply to every `json.dump` / `write_text` call in `storage.py`. There are approximately 8-10 write sites.

**Acceptance criteria**: Kill the process mid-write (e.g., `kill -9` during a commit) and verify the `.intentlog/` directory is not corrupted.

---

## Phase 1: Cut the Distractions (Week 2)

### 1.1 Remove integration modules

These modules integrate with products that should depend on IntentLog, not the other way around. The dependency direction is inverted.

**Delete entirely:**

| File | LOC | Reason |
|------|-----|--------|
| `src/intentlog/integrations/boundary_daemon.py` | 623 | Boundary Daemon should import IntentLog |
| `src/intentlog/integrations/boundary_siem.py` | 698 | SIEM should import IntentLog |
| `src/intentlog/integrations/llm_classifier.py` | 526 | Downstream app, not core |
| `src/intentlog/integrations/memory_vault.py` | 182 | Memory Vault should import IntentLog |
| `tests/test_integrations.py` | 111 | Tests for deleted code |
| `tests/test_llm_classifier.py` | 702 | Tests for deleted code |

**Dependency check**: Nothing in core imports from `integrations/`. Clean cut.

**Total removed**: ~2,842 LOC

### 1.2 Extract MP-02 to its own package

MP-02 ("proof of cognitive work") is a research protocol, not a feature of a version control tool. It has its own spec document (`mp-02-spec.md`, 40.6 KB).

**Delete from this repo:**

| File | LOC | Reason |
|------|-----|--------|
| `src/intentlog/mp02/signal.py` | 155 | Separate project |
| `src/intentlog/mp02/observer.py` | 462 | Separate project |
| `src/intentlog/mp02/segmentation.py` | 448 | Separate project |
| `src/intentlog/mp02/validator.py` | 474 | Separate project |
| `src/intentlog/mp02/receipt.py` | 398 | Separate project |
| `src/intentlog/mp02/ledger.py` | 524 | Separate project |
| `src/intentlog/cli/mp02.py` | ~200 | CLI for deleted code |
| `tests/test_mp02.py` | 660 | Tests for deleted code |

**Dependency check**: Nothing in core imports from `mp02/`. The only cross-reference is `mp02/receipt.py` mentioning "git_commit" as an artifact type — unused.

**Action**: Create a new repo `kase1111-hash/mp02-protocol` and move the code there. It can import `intentlog.core.Intent` as a dependency.

**Total removed**: ~3,321 LOC

### 1.3 Remove premature features

These solve problems that don't exist yet (no users, no scale, no production deployment).

**Delete entirely:**

| File | LOC | Reason |
|------|-----|--------|
| `src/intentlog/triggers.py` | 882 | Workflow orchestration, not version control |
| `src/intentlog/sufficiency.py` | 602 | Scoring quality before anyone writes intents |
| `src/intentlog/ratelimit.py` | 603 | No users hitting rate limits |
| `src/intentlog/privacy.py` | 1,030 | Full encryption/ACL system is premature |
| `src/intentlog/cli/privacy.py` | ~150 | CLI for deleted code |
| `tests/test_triggers.py` | 711 | Tests for deleted code |
| `tests/test_privacy.py` | 618 | Tests for deleted code |

**Dependency check**:
- `triggers.py` imports from `context.py` — but nothing imports from triggers. Clean cut.
- `sufficiency.py` imports from `core.py` — but only `cli/analytics.py` and `__init__.py` import from it. Remove those imports.
- `ratelimit.py` is imported by `__init__.py`, `llm/openai.py`, `llm/anthropic.py`. Remove from `__init__.py`; the LLM providers will be deferred anyway (see Phase 2).
- `privacy.py` is imported only by `__init__.py` and `cli/privacy.py`. Clean cut.

**Total removed**: ~4,596 LOC

### 1.4 Phase 1 totals

| Category | LOC Removed |
|----------|-------------|
| Integrations | 2,842 |
| MP-02 | 3,321 |
| Premature features | 4,596 |
| **Total** | **~10,759** |

**Remaining codebase**: ~10,000 LOC production + ~5,000 LOC tests

### 1.5 Clean up documentation

**Delete:**
- `Memory-Vault-Integration.md` — references deleted code
- `SECURITY_AUDIT.md` — auditing an alpha is premature
- `AUDIT_REPORT.md` — same
- `Your-Work-Isnt-Worthless.md` — motivational content, not software docs
- `INTEGRATION.md` — references deleted integrations
- Sections of README referencing removed features

**Keep:**
- `README.md` (update to reflect new scope)
- `Doctrine-of-intent.md` (philosophical foundation — good)
- `CONTRIBUTING.md` (update)
- `CHANGELOG.md` (update)
- `SECURITY.md` (update)
- `PRODUCTION_READINESS.md` (update)
- `docs/` MkDocs site (trim to match remaining features)

---

## Phase 2: Defer the Nice-to-Haves (Week 2-3)

These features are good ideas but shouldn't block shipping. Move behind feature flags or into `extras/`.

### 2.1 Make LLM features fully optional

Currently `semantic.py` (1,177 LOC) is imported at the package level. It powers semantic diffs and search but the core workflow (commit, log, search, branch) should work without any LLM.

**Actions:**
- Remove `semantic` imports from `__init__.py`
- Gate `ilog diff` and `ilog search --semantic` behind `intentlog[llm]` install check
- Keep keyword-based `ilog search` working without LLMs
- Move `src/intentlog/llm/` providers to optional: they already are in `pyproject.toml`, just enforce it in code
- Move `cli/formalize.py` behind the same gate
- Remove `tests/test_formalization.py` from default test run (mark with `@pytest.mark.llm`)

**Do NOT delete**: The semantic features are a key differentiator. Just don't make them load-bearing.

### 2.2 Make crypto features fully optional

`crypto.py` and `merkle.py` provide tamper-evident history. Good feature, not needed for v0.2.

**Actions:**
- Merkle chain linking in `storage.py` should be opt-in (config flag in `.intentlog/config.json`)
- `ilog keys generate` and `ilog chain verify` should check for `cryptography` and give a clear error
- Keep `merkle.py` (pure Python, no external deps) but make chain linking optional
- Move `crypto.py` behind `intentlog[crypto]` gate (already declared in pyproject.toml)

### 2.3 Defer analytics/metrics/export

These are "measure what you've built" tools. They need data first.

**Actions:**
- Remove from `__init__.py` (already done in Phase 0)
- Keep the code in place but don't expose in CLI help by default
- Gate behind `--advanced` flag or similar
- Mark tests with `@pytest.mark.analytics`

### 2.4 Defer context/decorator

The `@intent_logger` decorator and context propagation are powerful but complex (982 + 475 = 1,457 LOC). Manual `ilog commit` is the phase 1 workflow.

**Actions:**
- Remove from `__init__.py` (already done in Phase 0)
- Keep code in place for programmatic users
- Don't advertise in README until v0.3

---

## Phase 3: Build the Missing Core (Weeks 3-5)

This is the actual product work. After cutting and deferring, the core is:

```
src/intentlog/
├── core.py          (~109 LOC)  — Intent, IntentLog dataclasses
├── storage.py       (~900 LOC)  — persistence, branches, file ops
├── merkle.py        (~521 LOC)  — hash chain linking (optional)
├── audit.py         (~65 LOC)   — audit logging
├── logging.py       (~450 LOC)  — structured logging
├── validation.py    (~300 LOC)  — input validation
├── filelock.py      (~300 LOC)  — cross-platform file locking
├── schema.py        (~600 LOC)  — JSON schema
├── cli/
│   ├── core.py      (~580 LOC)  — init, commit, log, search, branch, merge, diff, status
│   └── utils.py     (~200 LOC)  — shared CLI helpers
└── __init__.py      (trimmed)
```

**~4,000 LOC core**. That's a focused product.

### 3.1 Build git integration (THE priority)

This is the killer feature that the evaluation identified as missing. `ilog commit` should work alongside `git commit`, not in isolation.

**Implement:**

1. **`ilog init` detects existing git repo**
   - If `.git/` exists, store `git_root` in `.intentlog/config.json`
   - Offer to add `.intentlog/` to `.gitignore` (or not — user's choice)

2. **`ilog commit` captures git context**
   - If in a git repo, automatically record:
     - Current git branch name
     - Current git HEAD commit hash
     - Staged files list (from `git diff --cached --name-only`)
   - Store in intent metadata as `git_context`
   - This creates the bidirectional link: every intent knows which git commit it's associated with

3. **`ilog log` shows git context**
   - Display git commit hash and branch alongside each intent
   - Support `ilog log --git-commit <hash>` to find intents for a specific commit

4. **Git hook integration (optional)**
   - `ilog hooks install` adds a `prepare-commit-msg` hook
   - Hook prepends the latest intent summary to the git commit message
   - Or: `post-commit` hook that prompts for intent if none was recorded
   - Keep this opt-in and clearly documented

5. **`ilog blame`** (new command)
   - Given a file, show the intent reasoning for each git commit that touched it
   - Bridges `git log --follow <file>` with IntentLog's reasoning history

**Acceptance criteria**: A developer can do `ilog commit "Choosing PostgreSQL over MongoDB because..."`, then `git add . && git commit`, and later `ilog search "database choice"` returns the reasoning with the associated git commit hash.

### 3.2 Improve the core CLI experience

**Polish:**
- `ilog status` should show pending intents, current branch, git context
- `ilog log` should be paginated and readable (not dump raw JSON)
- `ilog search` keyword search should work well without LLMs (substring + tag matching)
- `ilog diff main..feature` should work for basic text comparison without LLMs
- Error messages should be clear and actionable (not stack traces)
- Add `--json` flag to all commands for scripting

**Add:**
- `ilog show <intent-id>` — display a single intent with full metadata
- `ilog tag <intent-id> <tag>` — add tags for organization
- `ilog link <intent-id> <intent-id>` — explicitly link related intents

### 3.3 Write real tests for the core

**Target**: 90%+ coverage on `core.py`, `storage.py`, `merkle.py`, `cli/core.py`.

| Test Area | What to Test |
|-----------|-------------|
| Storage atomicity | Kill mid-write, verify no corruption |
| Branch operations | Create, switch, list, delete, merge |
| Intent CRUD | Create, read, update metadata, search |
| Merkle chains | Chain creation, verification, tamper detection |
| CLI integration | Full `init -> commit -> log -> search -> branch -> merge` workflow |
| Git integration | `ilog commit` with and without git context |
| Error handling | Invalid inputs, missing project, permission errors |
| Concurrency | Two processes writing simultaneously (file locking) |

### 3.4 Write a real example workflow

Replace `examples/basic_usage.py` with an end-to-end walkthrough:

```
examples/
├── quickstart.sh          # 5-minute terminal session
├── team_workflow.sh        # Multi-person branching scenario
└── git_integration.sh     # Side-by-side with git
```

These should be runnable scripts, not pseudocode.

---

## Phase 4: Ship and Learn (Week 6)

### 4.1 Update README for v0.2

- Cut the feature list to what actually works
- Lead with the git integration story
- Remove references to MP-02, SIEM, Boundary Daemon, Memory Vault
- Add a "What IntentLog is NOT" section to set expectations
- Keep the "Connected Repositories" section but note that integrations are separate packages

### 4.2 Update pyproject.toml

- Bump version to 0.2.0
- Remove optional dependency groups for deleted features
- Keep: `crypto`, `llm`, `dev`, `docs`
- Update entry points if CLI commands changed

### 4.3 Publish and get feedback

- Tag v0.2.0 release
- Write a clear changelog
- Find one team or individual to use it for a real project
- Their feedback determines Phase 5 priorities

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Cut integrations rather than move to `extras/` | Inverted dependency direction can't be fixed with reorganization — the consuming products should depend on IntentLog |
| Extract MP-02 rather than delete | The protocol has independent value as a research contribution |
| Keep merkle.py but make optional | Pure Python, no deps, and tamper-evident history is a genuine differentiator once the core works |
| Defer LLM features rather than cut | Semantic diff/search are the "wow" demo — just shouldn't be load-bearing |
| Prioritize git integration over everything | This is the adoption wedge — developers already use git, IntentLog must meet them there |
| Don't build a web UI | Tempting but premature. CLI-first, prove the workflow, then consider UI |

---

## Success Criteria for v0.2.0

- [x] `pip install intentlog && ilog init myproject && ilog commit "reasoning..."` works first try
- [x] All core tests pass (`pytest tests/ -v` — 320 passed, 30 skipped)
- [x] `ilog commit` captures git context when in a git repo
- [x] `ilog search` returns relevant results without LLMs
- [x] `ilog log` is readable without `| python -m json.tool`
- [x] Package loads without `cryptography` installed (`CRYPTO_AVAILABLE=False`, no crash)
- [x] Core codebase ~5,300 LOC (deferred modules add ~7,300 LOC but are not load-bearing)
- [ ] At least one person outside the author has used it and given feedback

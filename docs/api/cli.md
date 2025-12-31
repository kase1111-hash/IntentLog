# CLI Module

The CLI module provides the command-line interface for IntentLog.

## Module Structure

The CLI is organized into submodules by command group:

| Module | Commands |
|--------|----------|
| `cli.core` | init, commit, branch, log, search, audit, status, diff, merge, config |
| `cli.mp02` | observe, segment, receipt, ledger, verify |
| `cli.analytics` | export, analytics, metrics, sufficiency |
| `cli.crypto` | keys, chain |
| `cli.privacy` | privacy |
| `cli.formalize` | formalize |

## Main Entry Point

::: intentlog.cli.main
    options:
      show_root_heading: true

::: intentlog.cli.create_parser
    options:
      show_root_heading: true

## Core Commands

::: intentlog.cli.core
    options:
      show_root_heading: true
      members:
        - cmd_init
        - cmd_commit
        - cmd_branch
        - cmd_log
        - cmd_search
        - cmd_status
        - cmd_diff
        - cmd_merge
        - cmd_config
        - register_core_commands

## MP-02 Commands

::: intentlog.cli.mp02
    options:
      show_root_heading: true
      members:
        - cmd_observe
        - cmd_segment
        - cmd_receipt
        - cmd_ledger
        - cmd_verify
        - register_mp02_commands

## Analytics Commands

::: intentlog.cli.analytics
    options:
      show_root_heading: true
      members:
        - cmd_export
        - cmd_analytics
        - cmd_metrics
        - cmd_sufficiency
        - register_analytics_commands

## Crypto Commands

::: intentlog.cli.crypto
    options:
      show_root_heading: true
      members:
        - cmd_keys
        - cmd_chain
        - register_crypto_commands

## CLI Usage Examples

```bash
# Initialize project
ilog init my-project

# Create commits
ilog commit "Implementing feature X for requirement Y"
ilog commit --sign "Critical security fix"  # Signed commit

# View history
ilog log
ilog log --limit 20 --branch feature

# Search
ilog search "authentication"
ilog search --semantic "user login flow"  # LLM-powered

# Analytics
ilog analytics summary
ilog metrics all
ilog export --format jsonl --output intents.jsonl

# Cryptographic operations
ilog keys generate --name my-key
ilog chain verify
ilog chain proof --sequence 5
```

# Storage Module

The storage module provides persistent storage with Merkle chain integrity.

## IntentLogStorage

Main storage class for managing IntentLog projects.

::: intentlog.storage.IntentLogStorage
    options:
      show_root_heading: true
      members:
        - init_project
        - load_config
        - save_config
        - add_intent
        - add_chained_intent
        - load_intents
        - save_intents
        - search_intents
        - create_branch
        - switch_branch
        - list_branches
        - verify_chain
        - get_inclusion_proof

## ChainedIntent

An intent with Merkle chain linking.

::: intentlog.storage.ChainedIntent
    options:
      show_root_heading: true

## Configuration Classes

### ProjectConfig

::: intentlog.storage.ProjectConfig
    options:
      show_root_heading: true

### LLMSettings

::: intentlog.storage.LLMSettings
    options:
      show_root_heading: true

## Exceptions

::: intentlog.storage.ProjectNotFoundError
    options:
      show_root_heading: true

::: intentlog.storage.ProjectExistsError
    options:
      show_root_heading: true

::: intentlog.storage.BranchNotFoundError
    options:
      show_root_heading: true

## Usage Examples

### Project Initialization

```python
from intentlog.storage import IntentLogStorage

storage = IntentLogStorage()

# Initialize a new project
config = storage.init_project("my-project")
print(f"Project: {config.project_name}")
print(f"Location: {storage.intentlog_dir}")

# Create a chained intent with signature
chained = storage.add_chained_intent(
    name="Initial design",
    reasoning="Setting up the project architecture",
    sign=True
)
print(f"Chain hash: {chained.chain_hash}")
print(f"Sequence: {chained.sequence}")
```

### Chain Verification

```python
# Verify chain integrity
result = storage.verify_chain("main")

if result.valid:
    print(f"Chain valid! Root hash: {result.root_hash}")
else:
    print(f"Chain broken at sequence {result.broken_at}")
    for error in result.errors:
        print(f"  Error: {error}")
```

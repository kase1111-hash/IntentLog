# Core Module

The core module provides the fundamental data models for IntentLog.

## Intent

The `Intent` class represents a single unit of recorded reasoning.

::: intentlog.core.Intent
    options:
      show_root_heading: true
      members:
        - intent_id
        - intent_name
        - intent_reasoning
        - timestamp
        - metadata
        - parent_intent_id
        - trust_level
        - to_dict
        - validate

## IntentLog

The `IntentLog` class manages a collection of intents.

::: intentlog.core.IntentLog
    options:
      show_root_heading: true
      members:
        - add_intent
        - get_intent
        - get_intent_chain
        - search_intents
        - export_to_dict

## Usage Examples

### Creating an Intent

```python
from intentlog.core import Intent
from datetime import datetime

intent = Intent(
    intent_name="Implement caching layer",
    intent_reasoning="Adding Redis caching to reduce database load and improve response times",
    metadata={"category": "performance", "priority": "high"}
)

print(f"Intent ID: {intent.intent_id}")
print(f"Created at: {intent.timestamp}")
```

### Managing an IntentLog

```python
from intentlog.core import IntentLog

log = IntentLog(project_name="my-project")

# Add intents
intent = log.add_intent(
    name="Implement caching layer",
    reasoning="Adding Redis caching to reduce database load and improve response times",
    metadata={"category": "performance"},
)

# Search by content
results = log.search_intents("caching")

# Export to dictionary
data = log.export_to_dict()
```

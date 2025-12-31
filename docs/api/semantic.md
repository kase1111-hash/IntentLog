# Semantic Module

The semantic module provides LLM-powered semantic features.

## SemanticEngine

Main class for semantic operations.

::: intentlog.semantic.SemanticEngine
    options:
      show_root_heading: true
      members:
        - semantic_search
        - diff_branches
        - formalize
        - formalize_chain
        - formalize_from_search

## FormalizationType

Types of formalized output.

::: intentlog.semantic.FormalizationType
    options:
      show_root_heading: true

## FormalizedOutput

Result of formalization.

::: intentlog.semantic.FormalizedOutput
    options:
      show_root_heading: true

## LLM Providers

### LLMProvider (Abstract)

::: intentlog.llm.provider.LLMProvider
    options:
      show_root_heading: true
      members:
        - complete
        - embed
        - is_available

### OpenAI Provider

::: intentlog.llm.openai.OpenAIProvider
    options:
      show_root_heading: true

### Anthropic Provider

::: intentlog.llm.anthropic.AnthropicProvider
    options:
      show_root_heading: true

## Usage Examples

### Semantic Search

```python
from intentlog.semantic import SemanticEngine
from intentlog.llm.registry import get_provider

# Get LLM provider
provider = get_provider("openai")
engine = SemanticEngine(provider)

# Semantic search
results = engine.semantic_search(
    query="user authentication flow",
    intents=intents,
    top_k=5
)

for result in results:
    print(f"{result.score:.0%} - {result.intent.intent_name}")
```

### Branch Diff

```python
# Compare intent branches semantically
diffs = engine.diff_branches(
    intents_a=main_intents,
    intents_b=feature_intents,
    branch_a="main",
    branch_b="feature"
)

for diff in diffs:
    print(f"Change: {diff.summary}")
```

### Deferred Formalization

```python
from intentlog.semantic import FormalizationType

# Formalize intent into code
result = engine.formalize(
    intent,
    formalization_type=FormalizationType.CODE,
    language="python"
)

print(f"Generated {result.language} code:")
print(result.content)
print(f"Confidence: {result.confidence:.0%}")

# Formalize into rules
rules = engine.formalize(
    intent,
    formalization_type=FormalizationType.RULES
)
print(rules.content)
```

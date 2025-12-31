# Analytics Module

The analytics module provides intent analytics and doctrine metrics.

## IntentAnalytics

Analyzes intent patterns and statistics.

::: intentlog.analytics.IntentAnalytics
    options:
      show_root_heading: true
      members:
        - compute_latency_stats
        - compute_frequency_stats
        - compute_error_stats
        - get_trending_intents
        - get_bottlenecks
        - generate_report

## IntentMetrics

Computes doctrine metrics for intents.

::: intentlog.metrics.IntentMetrics
    options:
      show_root_heading: true
      members:
        - compute_intent_density
        - compute_information_density
        - compute_auditability
        - compute_fraud_resistance
        - get_all_metrics

## IntentExporter

Exports intents in various formats.

::: intentlog.export.IntentExporter
    options:
      show_root_heading: true
      members:
        - export
        - get_stats

## SufficiencyTest

Tests intent quality against criteria.

::: intentlog.sufficiency.run_sufficiency_test
    options:
      show_root_heading: true

## Usage Examples

### Analytics Report

```python
from intentlog.analytics import IntentAnalytics, generate_summary

analytics = IntentAnalytics(intents)

# Get latency statistics
latency = analytics.compute_latency_stats()
print(f"Mean latency: {latency.mean:.2f}ms")
print(f"P95 latency: {latency.p95:.2f}ms")

# Find bottlenecks
bottlenecks = analytics.get_bottlenecks(top_n=5)
for name, avg_latency in bottlenecks:
    print(f"{name}: {avg_latency:.2f}ms")

# Generate full summary
print(generate_summary(intents))
```

### Doctrine Metrics

```python
from intentlog.metrics import IntentMetrics

metrics = IntentMetrics(intents)

# Compute all metrics
all_metrics = metrics.get_all_metrics()

# Intent Density (Di)
density = all_metrics['intent_density']
print(f"Di: {density['score']:.3f}")

# Auditability
audit = all_metrics['auditability']
print(f"Auditability: {audit['score']:.3f} ({audit['rating']})")
```

### Exporting Intents

```python
from intentlog.export import IntentExporter, ExportFormat

exporter = IntentExporter(
    format_config=ExportFormat(format_type="jsonl")
)

# Export to file
exporter.export(intents, Path("intents.jsonl"))

# Export with anonymization
from intentlog.export import AnonymizationConfig

exporter = IntentExporter(
    anonymization=AnonymizationConfig()
)
exporter.export(intents, Path("anonymized.jsonl"))
```

"""
Analytics and Metrics CLI commands for IntentLog.

Commands: export, analytics, metrics, sufficiency
"""

import sys
from pathlib import Path
from datetime import datetime

from ..storage import IntentLogStorage, ProjectNotFoundError, BranchNotFoundError
from ..export import IntentExporter, ExportFilter, ExportFormat, AnonymizationConfig
from ..analytics import IntentAnalytics, generate_summary
from ..metrics import IntentMetrics
from ..sufficiency import run_sufficiency_test
from .utils import load_config_or_exit


def cmd_export(args):
    """Export intents for evaluation or fine-tuning"""
    format_type = getattr(args, 'format', 'jsonl')
    output = getattr(args, 'output', None)
    anonymize = getattr(args, 'anonymize', False)
    branch = getattr(args, 'branch', None)
    start_date = getattr(args, 'start', None)
    end_date = getattr(args, 'end', None)
    category = getattr(args, 'category', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    # Build filter config
    filter_config = ExportFilter()
    if start_date:
        filter_config.start_date = datetime.fromisoformat(start_date)
    if end_date:
        filter_config.end_date = datetime.fromisoformat(end_date)
    if category:
        filter_config.categories = [category]

    # Build format config
    format_config = ExportFormat(
        format_type=format_type,
        pretty_print=(format_type == 'json'),
    )

    # Build anonymization config
    anonymization = AnonymizationConfig() if anonymize else None

    # Create exporter
    exporter = IntentExporter(
        filter_config=filter_config,
        anonymization=anonymization,
        format_config=format_config,
    )

    # Export
    if output:
        output_path = Path(output)
        exporter.export(intents, output_path)
        stats = exporter.get_stats(intents)
        print(f"Exported {stats['filtered_intents']} intents to {output}")
        print(f"  Format: {format_type}")
        if anonymize:
            print(f"  Anonymized: Yes")
        if stats['filter_ratio'] < 1.0:
            print(f"  Filtered: {stats['total_intents']} -> {stats['filtered_intents']}")
    else:
        # Print to stdout
        result = exporter.export(intents)
        print(result)


def cmd_analytics(args):
    """Generate analytics report for intents"""
    action = args.action
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    analytics = IntentAnalytics(intents)

    if action == "summary" or action is None:
        # Generate full summary
        summary = generate_summary(intents)
        print(summary)

    elif action == "latency":
        # Show latency statistics
        stats = analytics.compute_latency_stats()
        print("Latency Statistics:")
        print(f"  Count: {stats.count}")
        print(f"  Mean: {stats.mean:.2f}ms")
        print(f"  Median: {stats.median:.2f}ms")
        print(f"  Std Dev: {stats.std_dev:.2f}ms")
        print(f"  Min: {stats.min}ms")
        print(f"  Max: {stats.max}ms")
        print(f"  P95: {stats.p95:.2f}ms")
        print(f"  P99: {stats.p99:.2f}ms")

    elif action == "frequency":
        # Show frequency statistics
        stats = analytics.compute_frequency_stats()
        print("Frequency Statistics:")
        print(f"  Total intents: {stats.total_count}")
        print(f"  Date range: {stats.date_range_days} days")
        print(f"  Intents per day: {stats.intents_per_day:.2f}")
        print(f"  Intents per hour: {stats.intents_per_hour:.2f}")
        print(f"  Peak hour: {stats.peak_hour}:00")
        print(f"  Peak day: {stats.peak_day}")
        print("\nTop categories:")
        for cat, count in list(stats.by_category.items())[:5]:
            print(f"    {cat}: {count}")

    elif action == "errors":
        # Show error statistics
        stats = analytics.compute_error_stats()
        print("Error Statistics:")
        print(f"  Total errors: {stats.total_errors}")
        print(f"  Error rate: {stats.error_rate * 100:.2f}%")
        if stats.errors_by_type:
            print("\nBy type:")
            for typ, count in stats.errors_by_type.items():
                print(f"    {typ}: {count}")

    elif action == "trends":
        # Show trending intents
        window = getattr(args, 'window', 7)
        top_n = getattr(args, 'top', 10)
        trends = analytics.get_trending_intents(window_days=window, top_n=top_n)
        print(f"Trending Intents (last {window} days):\n")
        for intent_name, count in trends:
            print(f"  {count:3d} x {intent_name}")

    elif action == "bottlenecks":
        # Show bottlenecks
        threshold = getattr(args, 'threshold', None)
        top_n = getattr(args, 'top', 10)
        bottlenecks = analytics.get_bottlenecks(
            latency_threshold_ms=threshold,
            top_n=top_n
        )
        print("Bottlenecks (high latency intents):\n")
        for intent_name, avg_latency in bottlenecks:
            print(f"  {avg_latency:7.2f}ms - {intent_name}")

    elif action == "report":
        # Generate full report
        report = analytics.generate_report()
        print("="*60)
        print("INTENT ANALYTICS REPORT")
        print("="*60)
        print(f"\nDate Range: {report.date_range['start']} to {report.date_range['end']}")
        print(f"Total Intents: {report.total_intents}")
        print(f"\nLatency: mean={report.latency.mean:.2f}ms, p95={report.latency.p95:.2f}ms")
        print(f"Frequency: {report.frequency.intents_per_day:.2f}/day")
        print(f"Errors: {report.errors.error_rate * 100:.2f}%")
        if report.activity:
            print(f"Sessions: {report.activity.session_count}")
        print("="*60)

    else:
        print(f"Unknown action: {action}")
        print("Available: summary, latency, frequency, errors, trends, bottlenecks, report")


def cmd_metrics(args):
    """Compute doctrine metrics for intents"""
    action = args.action
    branch = getattr(args, 'branch', None)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    metrics = IntentMetrics(intents)

    if action == "all" or action is None:
        # Show all metrics
        all_metrics = metrics.get_all_metrics()
        print("="*60)
        print("DOCTRINE METRICS")
        print("="*60)

        density = all_metrics['intent_density']
        print(f"\nIntent Density (Di): {density['score']:.3f}")
        print(f"  Resolution: {density['components']['resolution']:.3f}")
        print(f"  Continuity: {density['components']['continuity']:.3f}")
        print(f"  Coverage: {density['components']['coverage']:.3f}")

        info = all_metrics['information_density']
        print(f"\nInformation Density: {info['overall_score']:.3f}")
        print(f"  Content depth: {info['components']['content_depth']:.3f}")
        print(f"  Metadata richness: {info['components']['metadata_richness']:.3f}")
        print(f"  Avg reasoning words: {info['metrics']['avg_reasoning_words']:.1f}")

        audit = all_metrics['auditability']
        print(f"\nAuditability: {audit['score']:.3f} ({audit['rating']})")

        fraud = all_metrics['fraud_resistance']
        print(f"Fraud Resistance: {fraud['score']:.3f} ({fraud['rating']})")
        print("="*60)

    elif action == "density":
        # Show intent density
        density = metrics.compute_intent_density()
        print("Intent Density Metrics:")
        print(f"  Di: {density.Di:.3f}")
        print(f"  Resolution (R): {density.resolution:.3f}")
        print(f"  Continuity (C): {density.continuity:.3f}")
        print(f"  Coverage (Co): {density.coverage:.3f}")
        print(f"  Sample size: {density.sample_size}")

    elif action == "info":
        # Show information density
        info = metrics.compute_information_density()
        print("Information Density:")
        print(f"  Avg words: {info.avg_words:.1f}")
        print(f"  Avg chars: {info.avg_chars:.1f}")
        print(f"  Unique terms ratio: {info.unique_terms_ratio:.3f}")
        print(f"  Compression ratio: {info.compression_ratio:.3f}")
        print(f"  Entropy: {info.entropy:.3f}")

    elif action == "auditability":
        # Show auditability score
        audit = metrics.compute_auditability()
        print(f"Auditability Score: {audit.score:.3f}")
        print(f"Rating: {audit.rating}")
        print("\nComponents:")
        for comp, val in audit.components.items():
            print(f"  {comp}: {val:.3f}")

    elif action == "fraud":
        # Show fraud resistance
        fraud = metrics.compute_fraud_resistance()
        print(f"Fraud Resistance Score: {fraud.score:.3f}")
        print(f"Rating: {fraud.rating}")
        print("\nFactors:")
        for factor, val in fraud.factors.items():
            print(f"  {factor}: {val:.3f}")

    else:
        print(f"Unknown action: {action}")
        print("Available: all, density, info, auditability, fraud")


def cmd_sufficiency(args):
    """Run Intent Sufficiency Test"""
    branch = getattr(args, 'branch', None)
    author = getattr(args, 'author', None)
    verbose = getattr(args, 'verbose', False)

    storage = IntentLogStorage()

    try:
        config = storage.load_config()
        branch = branch or config.current_branch
        intents = storage.load_intents(branch)
    except ProjectNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BranchNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not intents:
        print(f"No intents on branch '{branch}'")
        return

    # Run sufficiency test
    report = run_sufficiency_test(intents, expected_author=author)

    print("="*60)
    print("INTENT SUFFICIENCY TEST")
    print("="*60)
    print(f"\nResult: {'PASS' if report.passed else 'FAIL'}")
    print(f"Score: {report.overall_score:.2f}/5.00")
    print(f"Criteria passed: {report.criteria_passed}/{report.total_criteria}")

    print("\nCriteria Results:")
    for criterion, result in report.criteria.items():
        status = "PASS" if result.passed else "FAIL"
        symbol = "+" if result.passed else "x"
        print(f"  {symbol} {criterion}: {result.score:.2f} - {status}")
        if verbose and not result.passed:
            for issue in result.issues:
                print(f"      - {issue}")

    if report.recommendations and verbose:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    print("="*60)
    sys.exit(0 if report.passed else 1)


def register_analytics_commands(subparsers):
    """Register analytics commands with the argument parser."""
    # export command
    export_parser = subparsers.add_parser("export", help="Export intents for eval/fine-tuning")
    export_parser.add_argument("--format", "-f", default="jsonl",
                               choices=["json", "jsonl", "csv", "huggingface", "openai"],
                               help="Output format (default: jsonl)")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--anonymize", "-a", action="store_true",
                               help="Anonymize exported data")
    export_parser.add_argument("--branch", "-b", help="Export from specific branch")
    export_parser.add_argument("--start", help="Filter: start date (ISO format)")
    export_parser.add_argument("--end", help="Filter: end date (ISO format)")
    export_parser.add_argument("--category", "-c", help="Filter by category")
    export_parser.set_defaults(func=cmd_export)

    # analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Generate analytics reports")
    analytics_parser.add_argument("action", nargs="?", default="summary",
                                  choices=["summary", "latency", "frequency", "errors",
                                           "trends", "bottlenecks", "report"],
                                  help="Analytics action (default: summary)")
    analytics_parser.add_argument("--branch", "-b", help="Analyze specific branch")
    analytics_parser.add_argument("--window", "-w", type=int, default=7,
                                  help="Time window in days for trends (default: 7)")
    analytics_parser.add_argument("--top", "-t", type=int, default=10,
                                  help="Number of top results (default: 10)")
    analytics_parser.add_argument("--threshold", type=int,
                                  help="Latency threshold in ms for bottlenecks")
    analytics_parser.set_defaults(func=cmd_analytics)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Compute doctrine metrics")
    metrics_parser.add_argument("action", nargs="?", default="all",
                                choices=["all", "density", "info", "auditability", "fraud"],
                                help="Metrics to compute (default: all)")
    metrics_parser.add_argument("--branch", "-b", help="Analyze specific branch")
    metrics_parser.set_defaults(func=cmd_metrics)

    # sufficiency command
    sufficiency_parser = subparsers.add_parser("sufficiency", help="Run Intent Sufficiency Test")
    sufficiency_parser.add_argument("--branch", "-b", help="Test specific branch")
    sufficiency_parser.add_argument("--author", "-a", help="Expected author for attribution")
    sufficiency_parser.add_argument("--verbose", "-v", action="store_true",
                                    help="Show detailed issues and recommendations")
    sufficiency_parser.set_defaults(func=cmd_sufficiency)

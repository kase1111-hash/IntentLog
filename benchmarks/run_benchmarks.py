#!/usr/bin/env python3
"""
IntentLog Performance Benchmark Runner

Measures performance of key operations:
- Intent creation and storage
- Chain operations (hashing, verification)
- Search operations (text and semantic)
- Analytics computation
- Export operations

Usage:
    python -m benchmarks.run_benchmarks [--iterations N] [--output FILE]
"""

import argparse
import json
import sys
import time
import statistics
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    ops_per_second: float


class Benchmark:
    """Benchmark runner for IntentLog operations."""

    def __init__(self, iterations: int = 100, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
        self.temp_dir = None

    def setup(self):
        """Create temporary directory for benchmarks."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="intentlog_bench_"))

    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def run(self, name: str, func: Callable, setup: Callable = None) -> BenchmarkResult:
        """Run a benchmark and collect timing data."""
        times = []

        # Warmup runs
        for _ in range(self.warmup):
            if setup:
                setup()
            func()

        # Timed runs
        for _ in range(self.iterations):
            if setup:
                setup()

            start = time.perf_counter()
            func()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        total = sum(times)
        mean = statistics.mean(times)
        median = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        ops_per_sec = 1000 / mean if mean > 0 else 0

        result = BenchmarkResult(
            name=name,
            iterations=self.iterations,
            total_time_ms=total,
            mean_time_ms=mean,
            median_time_ms=median,
            std_dev_ms=std_dev,
            min_time_ms=min_time,
            max_time_ms=max_time,
            ops_per_second=ops_per_sec,
        )

        self.results.append(result)
        return result

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("INTENTLOG PERFORMANCE BENCHMARKS")
        print("=" * 80)
        print(f"\nIterations per benchmark: {self.iterations}")
        print(f"Warmup iterations: {self.warmup}")
        print()

        # Header
        print(f"{'Benchmark':<40} {'Mean (ms)':<12} {'Median (ms)':<12} {'Ops/sec':<12}")
        print("-" * 80)

        for result in self.results:
            print(
                f"{result.name:<40} "
                f"{result.mean_time_ms:<12.3f} "
                f"{result.median_time_ms:<12.3f} "
                f"{result.ops_per_second:<12.1f}"
            )

        print("=" * 80)

    def export_json(self, filepath: Path):
        """Export results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "iterations": self.iterations,
            "warmup": self.warmup,
            "results": [asdict(r) for r in self.results],
        }
        filepath.write_text(json.dumps(data, indent=2))
        print(f"\nResults exported to: {filepath}")


def run_core_benchmarks(bench: Benchmark):
    """Run core module benchmarks."""
    from intentlog.core import Intent, IntentLog

    print("\n[Core Module Benchmarks]")

    # Benchmark: Intent creation
    def create_intent():
        Intent(
            intent_name="Test intent for benchmarking",
            intent_reasoning="This is a test reasoning with some content for benchmarking purposes.",
            metadata={"category": "test", "priority": "high"},
        )

    result = bench.run("Intent creation", create_intent)
    print(f"  Intent creation: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: IntentLog add
    log = IntentLog()

    def add_to_log():
        intent = Intent(
            intent_name="Benchmark intent",
            intent_reasoning="Reasoning for benchmark",
        )
        log.add(intent)

    result = bench.run("IntentLog.add", add_to_log)
    print(f"  IntentLog.add: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: Intent serialization
    intent = Intent(
        intent_name="Serialization test",
        intent_reasoning="Testing serialization performance",
        metadata={"key": "value"},
    )

    def serialize_intent():
        intent.to_dict()

    result = bench.run("Intent.to_dict", serialize_intent)
    print(f"  Intent.to_dict: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: Intent deserialization
    data = intent.to_dict()

    def deserialize_intent():
        Intent.from_dict(data)

    result = bench.run("Intent.from_dict", deserialize_intent)
    print(f"  Intent.from_dict: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")


def run_storage_benchmarks(bench: Benchmark):
    """Run storage module benchmarks."""
    from intentlog.storage import IntentLogStorage, compute_intent_hash
    from intentlog.core import Intent

    print("\n[Storage Module Benchmarks]")

    # Setup project
    project_dir = bench.temp_dir / "storage_bench"
    project_dir.mkdir(exist_ok=True)

    import os
    original_cwd = os.getcwd()
    os.chdir(project_dir)

    try:
        storage = IntentLogStorage()
        storage.init_project("benchmark-project", force=True)

        # Benchmark: add_intent
        counter = [0]

        def add_intent():
            counter[0] += 1
            storage.add_intent(
                name=f"Benchmark intent {counter[0]}",
                reasoning="Reasoning for storage benchmark test",
                metadata={"iteration": counter[0]},
            )

        result = bench.run("IntentLogStorage.add_intent", add_intent)
        print(f"  add_intent: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

        # Benchmark: load_intents
        def load_intents():
            storage.load_intents()

        result = bench.run("IntentLogStorage.load_intents", load_intents)
        print(f"  load_intents: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

        # Benchmark: search_intents
        def search_intents():
            storage.search_intents("benchmark")

        result = bench.run("IntentLogStorage.search_intents", search_intents)
        print(f"  search_intents: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

        # Benchmark: compute_intent_hash
        intent = Intent(
            intent_name="Hash test",
            intent_reasoning="Testing hash computation",
        )

        def compute_hash():
            compute_intent_hash(intent)

        result = bench.run("compute_intent_hash", compute_hash)
        print(f"  compute_intent_hash: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    finally:
        os.chdir(original_cwd)


def run_chain_benchmarks(bench: Benchmark):
    """Run Merkle chain benchmarks."""
    from intentlog.merkle import compute_hash, build_merkle_tree

    print("\n[Chain/Merkle Benchmarks]")

    # Benchmark: SHA-256 hash computation
    data = b"Sample data for hashing " * 10

    def hash_data():
        compute_hash(data)

    result = bench.run("SHA-256 hash", hash_data)
    print(f"  SHA-256 hash: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: Merkle tree building (10 items)
    items_10 = [f"item_{i}".encode() for i in range(10)]

    def build_tree_10():
        build_merkle_tree(items_10)

    result = bench.run("Merkle tree (10 items)", build_tree_10)
    print(f"  Merkle tree (10 items): {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: Merkle tree building (100 items)
    items_100 = [f"item_{i}".encode() for i in range(100)]

    def build_tree_100():
        build_merkle_tree(items_100)

    result = bench.run("Merkle tree (100 items)", build_tree_100)
    print(f"  Merkle tree (100 items): {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")

    # Benchmark: Merkle tree building (1000 items)
    items_1000 = [f"item_{i}".encode() for i in range(1000)]

    def build_tree_1000():
        build_merkle_tree(items_1000)

    result = bench.run("Merkle tree (1000 items)", build_tree_1000)
    print(f"  Merkle tree (1000 items): {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")


def run_analytics_benchmarks(bench: Benchmark):
    """Run analytics module benchmarks."""
    from intentlog.core import Intent
    from intentlog.analytics import IntentAnalytics
    from intentlog.metrics import IntentMetrics
    from datetime import timedelta
    import random

    print("\n[Analytics Benchmarks]")

    # Generate test intents
    intents = []
    base_time = datetime.now()
    for i in range(100):
        intent = Intent(
            intent_name=f"Test intent {i}",
            intent_reasoning=f"This is reasoning {i} with some words " * 5,
            metadata={
                "category": random.choice(["feature", "bugfix", "refactor"]),
                "latency_ms": random.randint(10, 500),
            },
        )
        # Adjust timestamp
        intent.timestamp = base_time - timedelta(hours=i)
        intents.append(intent)

    # Benchmark: Analytics initialization
    def create_analytics():
        IntentAnalytics(intents)

    result = bench.run("IntentAnalytics init (100 intents)", create_analytics)
    print(f"  Analytics init: {result.mean_time_ms:.3f}ms")

    analytics = IntentAnalytics(intents)

    # Benchmark: Latency stats
    def compute_latency():
        analytics.compute_latency_stats()

    result = bench.run("compute_latency_stats", compute_latency)
    print(f"  compute_latency_stats: {result.mean_time_ms:.3f}ms")

    # Benchmark: Frequency stats
    def compute_frequency():
        analytics.compute_frequency_stats()

    result = bench.run("compute_frequency_stats", compute_frequency)
    print(f"  compute_frequency_stats: {result.mean_time_ms:.3f}ms")

    # Benchmark: Metrics
    def create_metrics():
        IntentMetrics(intents)

    result = bench.run("IntentMetrics init (100 intents)", create_metrics)
    print(f"  Metrics init: {result.mean_time_ms:.3f}ms")

    metrics = IntentMetrics(intents)

    # Benchmark: Intent density
    def compute_density():
        metrics.compute_intent_density()

    result = bench.run("compute_intent_density", compute_density)
    print(f"  compute_intent_density: {result.mean_time_ms:.3f}ms")


def run_export_benchmarks(bench: Benchmark):
    """Run export module benchmarks."""
    from intentlog.core import Intent
    from intentlog.export import IntentExporter, ExportFormat

    print("\n[Export Benchmarks]")

    # Generate test intents
    intents = [
        Intent(
            intent_name=f"Export test {i}",
            intent_reasoning=f"Reasoning for export benchmark {i}",
            metadata={"index": i},
        )
        for i in range(100)
    ]

    # Benchmark: JSON export
    exporter_json = IntentExporter(
        format_config=ExportFormat(format_type="json", pretty_print=False)
    )

    def export_json():
        exporter_json.export(intents)

    result = bench.run("Export JSON (100 intents)", export_json)
    print(f"  Export JSON: {result.mean_time_ms:.3f}ms")

    # Benchmark: JSONL export
    exporter_jsonl = IntentExporter(
        format_config=ExportFormat(format_type="jsonl")
    )

    def export_jsonl():
        exporter_jsonl.export(intents)

    result = bench.run("Export JSONL (100 intents)", export_jsonl)
    print(f"  Export JSONL: {result.mean_time_ms:.3f}ms")

    # Benchmark: CSV export
    exporter_csv = IntentExporter(
        format_config=ExportFormat(format_type="csv")
    )

    def export_csv():
        exporter_csv.export(intents)

    result = bench.run("Export CSV (100 intents)", export_csv)
    print(f"  Export CSV: {result.mean_time_ms:.3f}ms")


def main():
    parser = argparse.ArgumentParser(description="Run IntentLog performance benchmarks")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: 10 iterations, 2 warmup"
    )

    args = parser.parse_args()

    iterations = 10 if args.quick else args.iterations
    warmup = 2 if args.quick else args.warmup

    bench = Benchmark(iterations=iterations, warmup=warmup)

    try:
        bench.setup()

        print(f"\nRunning IntentLog benchmarks...")
        print(f"Iterations: {iterations}, Warmup: {warmup}")

        # Run all benchmark suites
        run_core_benchmarks(bench)
        run_storage_benchmarks(bench)
        run_chain_benchmarks(bench)
        run_analytics_benchmarks(bench)
        run_export_benchmarks(bench)

        # Print summary
        bench.print_results()

        # Export if requested
        if args.output:
            bench.export_json(Path(args.output))

    finally:
        bench.teardown()


if __name__ == "__main__":
    main()

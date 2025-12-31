"""
Pytest-benchmark compatible benchmarks for IntentLog.

Run with:
    pytest benchmarks/test_benchmarks.py --benchmark-only
    pytest benchmarks/test_benchmarks.py --benchmark-json=benchmark_results.json
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta
import random

# Skip all tests if pytest-benchmark is not installed
pytest_benchmark_installed = True
try:
    import pytest_benchmark
except ImportError:
    pytest_benchmark_installed = False


# Fixtures

@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for project tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="intentlog_bench_"))
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_intents():
    """Generate sample intents for benchmarking."""
    from intentlog.core import Intent

    intents = []
    base_time = datetime.now()
    for i in range(100):
        intent = Intent(
            intent_name=f"Sample intent {i}",
            intent_reasoning=f"This is sample reasoning {i} with content " * 3,
            metadata={
                "category": random.choice(["feature", "bugfix", "refactor"]),
                "latency_ms": random.randint(10, 500),
            },
        )
        intent.timestamp = base_time - timedelta(hours=i)
        intents.append(intent)
    return intents


# Core Module Benchmarks

@pytest.mark.skipif(not pytest_benchmark_installed, reason="pytest-benchmark not installed")
class TestCoreBenchmarks:
    """Benchmarks for intentlog.core module."""

    def test_intent_creation(self, benchmark):
        """Benchmark Intent object creation."""
        from intentlog.core import Intent

        def create_intent():
            return Intent(
                intent_name="Benchmark intent",
                intent_reasoning="Reasoning for benchmark testing purposes",
                metadata={"category": "test"},
            )

        benchmark(create_intent)

    def test_intent_to_dict(self, benchmark):
        """Benchmark Intent serialization."""
        from intentlog.core import Intent

        intent = Intent(
            intent_name="Serialization test",
            intent_reasoning="Testing serialization performance",
            metadata={"key": "value"},
        )

        benchmark(intent.to_dict)

    def test_intent_from_dict(self, benchmark):
        """Benchmark Intent deserialization."""
        from intentlog.core import Intent

        data = {
            "intent_id": "test-id",
            "intent_name": "Test intent",
            "intent_reasoning": "Test reasoning",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"key": "value"},
        }

        benchmark(Intent.from_dict, data)

    def test_intentlog_add(self, benchmark):
        """Benchmark adding intents to IntentLog."""
        from intentlog.core import Intent, IntentLog

        log = IntentLog()

        def add_intent():
            intent = Intent(
                intent_name="Benchmark intent",
                intent_reasoning="Reasoning",
            )
            log.add(intent)

        benchmark(add_intent)

    def test_intentlog_search(self, benchmark, sample_intents):
        """Benchmark IntentLog search."""
        from intentlog.core import IntentLog

        log = IntentLog()
        for intent in sample_intents:
            log.add(intent)

        benchmark(log.search, "sample")


# Storage Module Benchmarks

@pytest.mark.skipif(not pytest_benchmark_installed, reason="pytest-benchmark not installed")
class TestStorageBenchmarks:
    """Benchmarks for intentlog.storage module."""

    def test_compute_intent_hash(self, benchmark):
        """Benchmark intent hash computation."""
        from intentlog.core import Intent
        from intentlog.storage import compute_intent_hash

        intent = Intent(
            intent_name="Hash benchmark",
            intent_reasoning="Testing hash computation performance",
        )

        benchmark(compute_intent_hash, intent)

    def test_add_intent(self, benchmark, temp_project_dir):
        """Benchmark adding intent to storage."""
        from intentlog.storage import IntentLogStorage

        storage = IntentLogStorage()
        storage.init_project("benchmark", force=True)

        counter = [0]

        def add():
            counter[0] += 1
            storage.add_intent(
                name=f"Benchmark {counter[0]}",
                reasoning="Benchmark reasoning",
            )

        benchmark(add)

    def test_load_intents(self, benchmark, temp_project_dir):
        """Benchmark loading intents from storage."""
        from intentlog.storage import IntentLogStorage

        storage = IntentLogStorage()
        storage.init_project("benchmark", force=True)

        # Add some intents first
        for i in range(50):
            storage.add_intent(name=f"Intent {i}", reasoning="Reasoning")

        benchmark(storage.load_intents)

    def test_search_intents(self, benchmark, temp_project_dir):
        """Benchmark searching intents in storage."""
        from intentlog.storage import IntentLogStorage

        storage = IntentLogStorage()
        storage.init_project("benchmark", force=True)

        # Add some intents first
        for i in range(50):
            storage.add_intent(name=f"Intent {i}", reasoning=f"Reasoning about topic {i}")

        benchmark(storage.search_intents, "topic")


# Merkle/Chain Benchmarks

@pytest.mark.skipif(not pytest_benchmark_installed, reason="pytest-benchmark not installed")
class TestMerkleBenchmarks:
    """Benchmarks for Merkle tree operations."""

    def test_sha256_hash(self, benchmark):
        """Benchmark SHA-256 hash computation."""
        from intentlog.merkle import compute_hash

        data = b"Sample data for hashing benchmark " * 10

        benchmark(compute_hash, data)

    def test_merkle_tree_10(self, benchmark):
        """Benchmark Merkle tree with 10 items."""
        from intentlog.merkle import build_merkle_tree

        items = [f"item_{i}".encode() for i in range(10)]

        benchmark(build_merkle_tree, items)

    def test_merkle_tree_100(self, benchmark):
        """Benchmark Merkle tree with 100 items."""
        from intentlog.merkle import build_merkle_tree

        items = [f"item_{i}".encode() for i in range(100)]

        benchmark(build_merkle_tree, items)

    def test_merkle_tree_1000(self, benchmark):
        """Benchmark Merkle tree with 1000 items."""
        from intentlog.merkle import build_merkle_tree

        items = [f"item_{i}".encode() for i in range(1000)]

        benchmark(build_merkle_tree, items)


# Analytics Benchmarks

@pytest.mark.skipif(not pytest_benchmark_installed, reason="pytest-benchmark not installed")
class TestAnalyticsBenchmarks:
    """Benchmarks for analytics module."""

    def test_analytics_init(self, benchmark, sample_intents):
        """Benchmark IntentAnalytics initialization."""
        from intentlog.analytics import IntentAnalytics

        benchmark(IntentAnalytics, sample_intents)

    def test_latency_stats(self, benchmark, sample_intents):
        """Benchmark latency statistics computation."""
        from intentlog.analytics import IntentAnalytics

        analytics = IntentAnalytics(sample_intents)

        benchmark(analytics.compute_latency_stats)

    def test_frequency_stats(self, benchmark, sample_intents):
        """Benchmark frequency statistics computation."""
        from intentlog.analytics import IntentAnalytics

        analytics = IntentAnalytics(sample_intents)

        benchmark(analytics.compute_frequency_stats)

    def test_metrics_density(self, benchmark, sample_intents):
        """Benchmark intent density computation."""
        from intentlog.metrics import IntentMetrics

        metrics = IntentMetrics(sample_intents)

        benchmark(metrics.compute_intent_density)

    def test_metrics_all(self, benchmark, sample_intents):
        """Benchmark all metrics computation."""
        from intentlog.metrics import IntentMetrics

        metrics = IntentMetrics(sample_intents)

        benchmark(metrics.get_all_metrics)


# Export Benchmarks

@pytest.mark.skipif(not pytest_benchmark_installed, reason="pytest-benchmark not installed")
class TestExportBenchmarks:
    """Benchmarks for export module."""

    def test_export_json(self, benchmark, sample_intents):
        """Benchmark JSON export."""
        from intentlog.export import IntentExporter, ExportFormat

        exporter = IntentExporter(
            format_config=ExportFormat(format_type="json", pretty_print=False)
        )

        benchmark(exporter.export, sample_intents)

    def test_export_jsonl(self, benchmark, sample_intents):
        """Benchmark JSONL export."""
        from intentlog.export import IntentExporter, ExportFormat

        exporter = IntentExporter(
            format_config=ExportFormat(format_type="jsonl")
        )

        benchmark(exporter.export, sample_intents)

    def test_export_csv(self, benchmark, sample_intents):
        """Benchmark CSV export."""
        from intentlog.export import IntentExporter, ExportFormat

        exporter = IntentExporter(
            format_config=ExportFormat(format_type="csv")
        )

        benchmark(exporter.export, sample_intents)

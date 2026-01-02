"""
Load Testing for IntentLog

This module provides load and stress tests for IntentLog to verify:
- Concurrent access handling
- File locking correctness
- Performance under load
- Memory usage patterns
- Chain integrity under stress
"""

import os
import sys
import tempfile
import threading
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest


# Skip entire module if running in CI without load test flag
LOAD_TEST_ENABLED = os.environ.get("INTENTLOG_LOAD_TEST", "0") == "1"
pytestmark = pytest.mark.skipif(
    not LOAD_TEST_ENABLED,
    reason="Load tests disabled. Set INTENTLOG_LOAD_TEST=1 to enable."
)


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    duration_seconds: float
    operations_per_second: float
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def summary(self) -> str:
        return f"""
Load Test: {self.name}
========================
Total Operations:    {self.total_operations}
Successful:          {self.successful_operations}
Failed:              {self.failed_operations}
Success Rate:        {self.success_rate:.2%}
Duration:            {self.duration_seconds:.2f}s
Throughput:          {self.operations_per_second:.2f} ops/sec

Latency (ms):
  Average:           {self.avg_latency_ms:.2f}
  P50:               {self.p50_latency_ms:.2f}
  P95:               {self.p95_latency_ms:.2f}
  P99:               {self.p99_latency_ms:.2f}

Errors: {len(self.errors)}
"""


class LoadTestHarness:
    """Harness for running load tests on IntentLog."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.results: List[LoadTestResult] = []

    def run_concurrent_test(
        self,
        name: str,
        operation: Callable[[], None],
        num_threads: int = 10,
        operations_per_thread: int = 100,
        warmup_operations: int = 10,
    ) -> LoadTestResult:
        """
        Run a concurrent load test.

        Args:
            name: Test name
            operation: Operation to execute
            num_threads: Number of concurrent threads
            operations_per_thread: Operations each thread performs
            warmup_operations: Warmup iterations (not measured)

        Returns:
            LoadTestResult with metrics
        """
        # Warmup
        for _ in range(warmup_operations):
            try:
                operation()
            except Exception:
                pass

        total_operations = num_threads * operations_per_thread
        successful = 0
        failed = 0
        latencies = []
        errors = []
        lock = threading.Lock()

        def worker():
            nonlocal successful, failed
            local_latencies = []
            local_errors = []

            for _ in range(operations_per_thread):
                start = time.perf_counter()
                try:
                    operation()
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    local_latencies.append(elapsed)
                    with lock:
                        successful += 1
                except Exception as e:
                    with lock:
                        failed += 1
                    local_errors.append(str(e))

            with lock:
                latencies.extend(local_latencies)
                errors.extend(local_errors)

        start_time = time.perf_counter()

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.perf_counter() - start_time
        ops_per_sec = total_operations / duration if duration > 0 else 0

        result = LoadTestResult(
            name=name,
            total_operations=total_operations,
            successful_operations=successful,
            failed_operations=failed,
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            latencies_ms=latencies,
            errors=errors[:10],  # Limit error storage
        )

        self.results.append(result)
        return result


@pytest.fixture
def temp_project():
    """Create a temporary IntentLog project for testing."""
    from intentlog.storage import IntentLogStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        storage = IntentLogStorage(project_path)
        storage.init_project("load-test-project")
        yield project_path


@pytest.fixture
def harness(temp_project):
    """Create a LoadTestHarness for the temporary project."""
    return LoadTestHarness(temp_project)


class TestConcurrentWrites:
    """Test concurrent write operations."""

    def test_concurrent_intent_creation(self, temp_project, harness):
        """Test creating intents from multiple threads."""
        from intentlog.storage import IntentLogStorage

        counter = [0]
        lock = threading.Lock()

        def create_intent():
            storage = IntentLogStorage(temp_project)
            with lock:
                counter[0] += 1
                n = counter[0]
            storage.add_intent(
                name=f"Load Test Intent {n}",
                reasoning=f"Testing concurrent access with intent number {n}",
                metadata={"test": True, "thread": threading.current_thread().name},
            )

        result = harness.run_concurrent_test(
            name="Concurrent Intent Creation",
            operation=create_intent,
            num_threads=5,
            operations_per_thread=20,
        )

        print(result.summary())

        # Verify all intents were created
        storage = IntentLogStorage(temp_project)
        intents = storage.load_intents()

        assert result.success_rate >= 0.95, f"Too many failures: {result.failed_operations}"
        assert len(intents) == result.successful_operations
        assert result.operations_per_second > 10, "Throughput too low"

    def test_concurrent_branch_operations(self, temp_project, harness):
        """Test creating and switching branches concurrently."""
        from intentlog.storage import IntentLogStorage

        branch_counter = [0]
        lock = threading.Lock()

        def branch_operation():
            storage = IntentLogStorage(temp_project)
            with lock:
                branch_counter[0] += 1
                n = branch_counter[0]

            # Create branch
            branch_name = f"test-branch-{n}"
            storage.create_branch(branch_name)

            # Add intent to branch
            storage.add_intent(
                name=f"Intent on {branch_name}",
                reasoning="Testing branch operations",
                branch=branch_name,
            )

        result = harness.run_concurrent_test(
            name="Concurrent Branch Operations",
            operation=branch_operation,
            num_threads=3,
            operations_per_thread=10,
        )

        print(result.summary())

        # Verify branches were created
        storage = IntentLogStorage(temp_project)
        branches = storage.list_branches()

        assert result.success_rate >= 0.90, f"Too many failures: {result.failed_operations}"
        assert len(branches) >= result.successful_operations


class TestConcurrentReads:
    """Test concurrent read operations."""

    def test_concurrent_intent_reads(self, temp_project, harness):
        """Test reading intents from multiple threads."""
        from intentlog.storage import IntentLogStorage

        # Setup: Create some intents first
        storage = IntentLogStorage(temp_project)
        for i in range(50):
            storage.add_intent(
                name=f"Pre-existing Intent {i}",
                reasoning=f"Setup intent number {i}",
            )

        def read_intents():
            storage = IntentLogStorage(temp_project)
            intents = storage.load_intents()
            assert len(intents) >= 50

        result = harness.run_concurrent_test(
            name="Concurrent Intent Reads",
            operation=read_intents,
            num_threads=10,
            operations_per_thread=50,
        )

        print(result.summary())

        assert result.success_rate == 1.0, "Reads should not fail"
        assert result.operations_per_second > 100, "Read throughput too low"

    def test_concurrent_search(self, temp_project, harness):
        """Test searching intents from multiple threads."""
        from intentlog.storage import IntentLogStorage

        # Setup: Create searchable intents
        storage = IntentLogStorage(temp_project)
        keywords = ["architecture", "design", "implementation", "testing", "deployment"]
        for i in range(100):
            keyword = keywords[i % len(keywords)]
            storage.add_intent(
                name=f"{keyword.title()} Decision {i}",
                reasoning=f"This is about {keyword} considerations for component {i}",
            )

        def search_intents():
            storage = IntentLogStorage(temp_project)
            keyword = random.choice(keywords)
            results = storage.search_intents(keyword)
            assert len(results) > 0

        result = harness.run_concurrent_test(
            name="Concurrent Search",
            operation=search_intents,
            num_threads=8,
            operations_per_thread=25,
        )

        print(result.summary())

        assert result.success_rate == 1.0, "Searches should not fail"


class TestMixedWorkload:
    """Test mixed read/write workloads."""

    def test_mixed_operations(self, temp_project, harness):
        """Test realistic mixed workload."""
        from intentlog.storage import IntentLogStorage

        counter = [0]
        lock = threading.Lock()

        def mixed_operation():
            storage = IntentLogStorage(temp_project)

            # 70% reads, 20% writes, 10% searches
            op = random.random()

            if op < 0.70:
                # Read
                storage.load_intents()
            elif op < 0.90:
                # Write
                with lock:
                    counter[0] += 1
                    n = counter[0]
                storage.add_intent(
                    name=f"Mixed Workload Intent {n}",
                    reasoning="Created during mixed workload test",
                )
            else:
                # Search
                storage.search_intents("workload")

        result = harness.run_concurrent_test(
            name="Mixed Workload",
            operation=mixed_operation,
            num_threads=6,
            operations_per_thread=100,
        )

        print(result.summary())

        assert result.success_rate >= 0.95, f"Too many failures: {result.failed_operations}"


class TestFileLocking:
    """Test file locking correctness under stress."""

    def test_no_data_corruption(self, temp_project, harness):
        """Verify no data corruption under concurrent writes."""
        from intentlog.storage import IntentLogStorage

        counter = [0]
        lock = threading.Lock()

        def write_with_verification():
            storage = IntentLogStorage(temp_project)
            with lock:
                counter[0] += 1
                n = counter[0]

            # Create intent with unique identifier
            unique_marker = f"UNIQUE_{n}_{time.time_ns()}"
            storage.add_intent(
                name=f"Integrity Test {n}",
                reasoning=f"Marker: {unique_marker}",
            )

        result = harness.run_concurrent_test(
            name="Data Integrity Under Load",
            operation=write_with_verification,
            num_threads=8,
            operations_per_thread=25,
        )

        print(result.summary())

        # Verify all data is intact
        storage = IntentLogStorage(temp_project)
        intents = storage.load_intents()

        # Each intent should have unique marker
        markers = set()
        for intent in intents:
            if "Marker:" in intent.intent_reasoning:
                marker = intent.intent_reasoning.split("Marker:")[1].strip()
                assert marker not in markers, f"Duplicate marker found: {marker}"
                markers.add(marker)

        assert len(markers) == result.successful_operations


class TestChainIntegrity:
    """Test Merkle chain integrity under load."""

    def test_chain_remains_valid(self, temp_project, harness):
        """Verify chain integrity is maintained under concurrent writes."""
        from intentlog.storage import IntentLogStorage

        counter = [0]
        lock = threading.Lock()

        def add_chained_intent():
            storage = IntentLogStorage(temp_project)
            with lock:
                counter[0] += 1
                n = counter[0]

            storage.add_chained_intent(
                name=f"Chained Intent {n}",
                reasoning=f"Testing chain integrity under load",
            )

        result = harness.run_concurrent_test(
            name="Chain Integrity Under Load",
            operation=add_chained_intent,
            num_threads=4,
            operations_per_thread=25,
        )

        print(result.summary())

        # Verify chain integrity
        storage = IntentLogStorage(temp_project)
        verification = storage.verify_chain()

        assert verification.valid, f"Chain integrity violated: {verification.error}"
        assert result.success_rate >= 0.90


class TestBackupUnderLoad:
    """Test backup operations under load."""

    def test_backup_during_writes(self, temp_project, harness):
        """Test backup creation while writes are happening."""
        from intentlog.storage import IntentLogStorage
        from intentlog.backup import BackupManager

        write_counter = [0]
        backup_counter = [0]
        lock = threading.Lock()

        def write_or_backup():
            # 90% writes, 10% backups
            if random.random() < 0.90:
                storage = IntentLogStorage(temp_project)
                with lock:
                    write_counter[0] += 1
                    n = write_counter[0]
                storage.add_intent(
                    name=f"Intent During Backup {n}",
                    reasoning="Created while backups may be running",
                )
            else:
                with lock:
                    backup_counter[0] += 1
                manager = BackupManager(temp_project)
                manager.create_backup(compress=False)

        result = harness.run_concurrent_test(
            name="Backup During Writes",
            operation=write_or_backup,
            num_threads=4,
            operations_per_thread=50,
        )

        print(result.summary())
        print(f"Writes: {write_counter[0]}, Backups: {backup_counter[0]}")

        assert result.success_rate >= 0.90


class TestPerformanceBenchmarks:
    """Performance benchmarks for tracking regressions."""

    def test_single_thread_throughput(self, temp_project):
        """Benchmark single-threaded write throughput."""
        from intentlog.storage import IntentLogStorage

        storage = IntentLogStorage(temp_project)
        num_operations = 100

        start = time.perf_counter()
        for i in range(num_operations):
            storage.add_intent(
                name=f"Benchmark Intent {i}",
                reasoning=f"Benchmarking single-threaded performance",
            )
        duration = time.perf_counter() - start

        ops_per_sec = num_operations / duration
        avg_latency_ms = (duration / num_operations) * 1000

        print(f"\nSingle-Thread Benchmark:")
        print(f"  Operations: {num_operations}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Avg Latency: {avg_latency_ms:.2f}ms")

        # Performance assertions (adjust based on baseline)
        assert ops_per_sec > 50, f"Single-thread throughput too low: {ops_per_sec}"
        assert avg_latency_ms < 50, f"Average latency too high: {avg_latency_ms}ms"

    def test_read_throughput(self, temp_project):
        """Benchmark read throughput."""
        from intentlog.storage import IntentLogStorage

        storage = IntentLogStorage(temp_project)

        # Setup
        for i in range(100):
            storage.add_intent(name=f"Setup {i}", reasoning="Setup")

        num_reads = 500

        start = time.perf_counter()
        for _ in range(num_reads):
            storage.load_intents()
        duration = time.perf_counter() - start

        ops_per_sec = num_reads / duration

        print(f"\nRead Benchmark:")
        print(f"  Operations: {num_reads}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")

        assert ops_per_sec > 100, f"Read throughput too low: {ops_per_sec}"


def run_all_load_tests():
    """Run all load tests and generate report."""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])


if __name__ == "__main__":
    os.environ["INTENTLOG_LOAD_TEST"] = "1"
    run_all_load_tests()

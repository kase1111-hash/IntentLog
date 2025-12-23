"""Tests for Phase 4: Analytics and Metrics"""
import pytest
from datetime import datetime, timedelta
import json

from intentlog.core import Intent
from intentlog.export import (
    IntentExporter, ExportFilter, ExportFormat, AnonymizationConfig,
    export_for_eval, export_for_finetuning
)
from intentlog.analytics import (
    IntentAnalytics, LatencyStats, FrequencyStats, ErrorStats,
    ActivityPattern, AnalyticsReport, generate_summary
)
from intentlog.metrics import (
    IntentMetrics, IntentDensity, InformationDensity,
    AuditabilityScore, FraudResistance
)
from intentlog.sufficiency import (
    SufficiencyTest, SufficiencyReport, SufficiencyResult,
    CriterionResult, run_sufficiency_test
)


# ============ Test Fixtures ============

@pytest.fixture
def sample_intents():
    """Create sample intents for testing"""
    base_time = datetime.now() - timedelta(hours=5)
    intents = []

    for i in range(10):
        intent = Intent(
            intent_name=f"Test Intent {i}",
            intent_reasoning=f"This is the reasoning for intent {i}. We need to accomplish this goal.",
            metadata={
                "category": "testing" if i % 2 == 0 else "development",
                "latency_ms": 100 + i * 50,
                "author": "test_user",
                "tags": ["test", "sample"],
            }
        )
        # Set timestamp with spacing
        intent.timestamp = base_time + timedelta(minutes=i * 10)
        intents.append(intent)

    return intents


@pytest.fixture
def minimal_intents():
    """Create minimal intents for edge cases"""
    return [
        Intent(
            intent_name="Short",
            intent_reasoning="Brief"
        )
    ]


# ============ Export Tests ============

class TestExportFilter:
    def test_basic_filter(self, sample_intents):
        """Test basic filtering matches all"""
        filter_config = ExportFilter()
        for intent in sample_intents:
            assert filter_config.matches(intent) is True

    def test_date_filter(self, sample_intents):
        """Test date range filtering"""
        # Use dates relative to the sample intents
        first_ts = sample_intents[0].timestamp
        last_ts = sample_intents[-1].timestamp
        mid_ts = first_ts + (last_ts - first_ts) / 2

        filter_config = ExportFilter(
            start_date=mid_ts,
            end_date=last_ts + timedelta(hours=1)
        )

        matches = [i for i in sample_intents if filter_config.matches(i)]
        assert len(matches) > 0
        assert len(matches) < len(sample_intents)

    def test_category_filter(self, sample_intents):
        """Test category filtering"""
        filter_config = ExportFilter(categories=["testing"])

        matches = [i for i in sample_intents if filter_config.matches(i)]
        assert all(i.metadata.get("category") == "testing" for i in matches)

    def test_name_pattern_filter(self, sample_intents):
        """Test regex pattern filtering"""
        filter_config = ExportFilter(name_pattern=r"Intent [0-2]")

        matches = [i for i in sample_intents if filter_config.matches(i)]
        assert len(matches) == 3  # Intent 0, 1, 2

    def test_latency_filter(self, sample_intents):
        """Test latency threshold filtering"""
        filter_config = ExportFilter(min_latency_ms=200, max_latency_ms=400)

        matches = [i for i in sample_intents if filter_config.matches(i)]
        for match in matches:
            latency = match.metadata.get("latency_ms")
            assert 200 <= latency <= 400


class TestAnonymization:
    def test_basic_anonymization(self, sample_intents):
        """Test basic anonymization"""
        config = AnonymizationConfig()
        anonymized = config.anonymize_intent(sample_intents[0], 0)

        assert anonymized["intent_name"] == "Intent_0000"
        assert anonymized["intent_id"] != sample_intents[0].intent_id

    def test_hash_ids(self, sample_intents):
        """Test ID hashing"""
        config = AnonymizationConfig(hash_ids=True)
        anon1 = config.anonymize_intent(sample_intents[0], 0)
        anon2 = config.anonymize_intent(sample_intents[0], 0)

        # Same intent should produce same hash
        assert anon1["intent_id"] == anon2["intent_id"]

    def test_timestamp_rounding(self, sample_intents):
        """Test timestamp rounding"""
        config = AnonymizationConfig(round_timestamps="day")
        anonymized = config.anonymize_intent(sample_intents[0], 0)

        # Should be start of day
        ts = datetime.fromisoformat(anonymized["timestamp"])
        assert ts.hour == 0
        assert ts.minute == 0

    def test_remove_metadata_keys(self, sample_intents):
        """Test metadata key removal"""
        config = AnonymizationConfig(remove_metadata_keys=["author"])
        anonymized = config.anonymize_intent(sample_intents[0], 0)

        assert "author" not in anonymized["metadata"]
        assert "category" in anonymized["metadata"]


class TestIntentExporter:
    def test_json_export(self, sample_intents):
        """Test JSON format export"""
        format_config = ExportFormat(format_type="json", pretty_print=True)
        exporter = IntentExporter(format_config=format_config)

        result = exporter.export(sample_intents)
        data = json.loads(result)

        assert isinstance(data, list)
        assert len(data) == len(sample_intents)

    def test_jsonl_export(self, sample_intents):
        """Test JSONL format export"""
        format_config = ExportFormat(format_type="jsonl")
        exporter = IntentExporter(format_config=format_config)

        result = exporter.export(sample_intents)
        lines = result.strip().split("\n")

        assert len(lines) == len(sample_intents)
        for line in lines:
            json.loads(line)  # Should be valid JSON

    def test_csv_export(self, sample_intents):
        """Test CSV format export"""
        format_config = ExportFormat(format_type="csv")
        exporter = IntentExporter(format_config=format_config)

        result = exporter.export(sample_intents)
        lines = result.strip().split("\n")

        # First line is header
        assert len(lines) == len(sample_intents) + 1
        assert "intent_name" in lines[0]

    def test_openai_export(self, sample_intents):
        """Test OpenAI fine-tuning format"""
        format_config = ExportFormat(
            format_type="openai",
            system_prompt="You analyze intents."
        )
        exporter = IntentExporter(format_config=format_config)

        result = exporter.export(sample_intents)
        lines = result.strip().split("\n")

        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) >= 2

    def test_export_with_filter(self, sample_intents):
        """Test export with filtering"""
        filter_config = ExportFilter(categories=["testing"])
        exporter = IntentExporter(filter_config=filter_config)

        result = exporter.export(sample_intents)
        data = [json.loads(line) for line in result.strip().split("\n")]

        assert len(data) < len(sample_intents)

    def test_get_stats(self, sample_intents):
        """Test export statistics"""
        filter_config = ExportFilter(categories=["testing"])
        exporter = IntentExporter(filter_config=filter_config)

        stats = exporter.get_stats(sample_intents)

        assert stats["total_intents"] == len(sample_intents)
        assert stats["filtered_intents"] < stats["total_intents"]
        assert 0 < stats["filter_ratio"] < 1


# ============ Analytics Tests ============

class TestLatencyStats:
    def test_compute_latency_stats(self, sample_intents):
        """Test latency statistics computation"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_latency_stats()

        assert stats.count == len(sample_intents)
        assert stats.mean > 0
        assert stats.median > 0
        assert stats.min <= stats.mean <= stats.max

    def test_latency_aliases(self, sample_intents):
        """Test latency property aliases"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_latency_stats()

        assert stats.mean == stats.mean_ms
        assert stats.median == stats.median_ms
        assert stats.p95 == stats.p95_ms

    def test_empty_latency(self, minimal_intents):
        """Test latency with no latency data"""
        analytics = IntentAnalytics(minimal_intents)
        stats = analytics.compute_latency_stats()

        assert stats.count == 0


class TestFrequencyStats:
    def test_compute_frequency_stats(self, sample_intents):
        """Test frequency statistics computation"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_frequency_stats()

        assert stats.total_count == len(sample_intents)
        assert len(stats.by_category) > 0
        assert stats.intents_per_day > 0

    def test_peak_detection(self, sample_intents):
        """Test peak hour and day detection"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_frequency_stats()

        assert isinstance(stats.peak_hour, int)
        assert 0 <= stats.peak_hour <= 23
        assert stats.peak_day != ""


class TestErrorStats:
    def test_compute_error_stats(self, sample_intents):
        """Test error statistics computation"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_error_stats()

        assert stats.total_errors >= 0
        assert 0 <= stats.error_rate <= 1

    def test_errors_by_type_alias(self, sample_intents):
        """Test errors_by_type alias"""
        analytics = IntentAnalytics(sample_intents)
        stats = analytics.compute_error_stats()

        assert stats.errors_by_type == stats.error_categories


class TestActivityPattern:
    def test_compute_activity_pattern(self, sample_intents):
        """Test activity pattern computation"""
        analytics = IntentAnalytics(sample_intents)
        pattern = analytics.compute_activity_pattern()

        assert pattern.session_count > 0
        assert len(pattern.active_hours) > 0
        assert pattern.avg_intents_per_day >= 0


class TestAnalyticsReport:
    def test_generate_report(self, sample_intents):
        """Test full report generation"""
        analytics = IntentAnalytics(sample_intents)
        report = analytics.generate_report()

        assert report.intent_count == len(sample_intents)
        assert report.total_intents == len(sample_intents)
        assert report.latency is not None
        assert report.frequency is not None

    def test_report_date_range(self, sample_intents):
        """Test report date range"""
        analytics = IntentAnalytics(sample_intents)
        report = analytics.generate_report()

        date_range = report.date_range
        assert "start" in date_range
        assert "end" in date_range

    def test_report_to_dict(self, sample_intents):
        """Test report serialization"""
        analytics = IntentAnalytics(sample_intents)
        report = analytics.generate_report()

        data = report.to_dict()
        assert "intent_count" in data
        assert "latency" in data

    def test_report_to_json(self, sample_intents):
        """Test JSON export"""
        analytics = IntentAnalytics(sample_intents)
        report = analytics.generate_report()

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["intent_count"] == len(sample_intents)


class TestTrendsAndBottlenecks:
    def test_get_trending_intents(self, sample_intents):
        """Test trending intents detection"""
        analytics = IntentAnalytics(sample_intents)
        trends = analytics.get_trending_intents(window_days=30)

        assert isinstance(trends, list)
        # Each item is (name, count, trend_score)
        if trends:
            assert len(trends[0]) == 3

    def test_get_bottlenecks(self, sample_intents):
        """Test bottleneck detection"""
        analytics = IntentAnalytics(sample_intents)
        bottlenecks = analytics.get_bottlenecks(latency_threshold_ms=100)

        assert isinstance(bottlenecks, list)


class TestGenerateSummary:
    def test_generate_summary(self, sample_intents):
        """Test summary generation"""
        summary = generate_summary(sample_intents)

        assert "IntentLog Analytics Summary" in summary
        assert "Total Intents" in summary


# ============ Metrics Tests ============

class TestIntentDensity:
    def test_compute_intent_density(self, sample_intents):
        """Test intent density computation"""
        metrics = IntentMetrics(sample_intents)
        density = metrics.compute_intent_density()

        assert 0 <= density.Di <= 1
        assert 0 <= density.resolution <= 1
        assert 0 <= density.continuity <= 1
        assert 0 <= density.coverage <= 1

    def test_density_to_dict(self, sample_intents):
        """Test density serialization"""
        metrics = IntentMetrics(sample_intents)
        density = metrics.compute_intent_density()

        data = density.to_dict()
        assert "score" in data
        assert "components" in data
        assert "resolution" in data["components"]


class TestInformationDensity:
    def test_compute_information_density(self, sample_intents):
        """Test information density computation"""
        metrics = IntentMetrics(sample_intents)
        info = metrics.compute_information_density()

        assert info.avg_words > 0
        assert info.avg_chars > 0
        assert 0 <= info.unique_terms_ratio <= 1


class TestAuditabilityScore:
    def test_compute_auditability(self, sample_intents):
        """Test auditability score computation"""
        metrics = IntentMetrics(sample_intents)
        audit = metrics.compute_auditability()

        assert 0 <= audit.score <= 1
        assert audit.rating in ["excellent", "good", "fair", "poor"]
        assert len(audit.components) > 0


class TestFraudResistance:
    def test_compute_fraud_resistance(self, sample_intents):
        """Test fraud resistance computation"""
        metrics = IntentMetrics(sample_intents)
        fraud = metrics.compute_fraud_resistance()

        assert 0 <= fraud.score <= 1
        assert fraud.rating in ["excellent", "good", "fair", "poor"]
        assert len(fraud.factors) > 0


class TestAllMetrics:
    def test_get_all_metrics(self, sample_intents):
        """Test getting all metrics"""
        metrics = IntentMetrics(sample_intents)
        all_metrics = metrics.get_all_metrics()

        assert "intent_density" in all_metrics
        assert "information_density" in all_metrics
        assert "auditability" in all_metrics
        assert "fraud_resistance" in all_metrics


# ============ Sufficiency Tests ============

class TestSufficiencyTest:
    def test_insufficient_data(self, minimal_intents):
        """Test with insufficient data"""
        report = run_sufficiency_test(minimal_intents)

        assert report.result == SufficiencyResult.INSUFFICIENT_DATA

    def test_sufficiency_test_runs(self, sample_intents):
        """Test that sufficiency test runs"""
        report = run_sufficiency_test(sample_intents)

        assert report.result in [
            SufficiencyResult.PASS,
            SufficiencyResult.PARTIAL,
            SufficiencyResult.FAIL,
        ]
        assert report.criteria_passed >= 0
        assert report.total_criteria == 5

    def test_sufficiency_criteria(self, sample_intents):
        """Test individual criteria"""
        test = SufficiencyTest(sample_intents)
        report = test.run()

        criteria = report.criteria
        assert "Continuity" in criteria
        assert "Directionality" in criteria
        assert "Resolution" in criteria
        assert "Temporal Anchoring" in criteria
        assert "Human Attribution" in criteria

    def test_criterion_result(self, sample_intents):
        """Test criterion result structure"""
        test = SufficiencyTest(sample_intents)
        report = test.run()

        for name, result in report.criteria.items():
            assert isinstance(result, CriterionResult)
            assert 0 <= result.score <= 1
            assert isinstance(result.passed, bool)


class TestSufficiencyReport:
    def test_report_properties(self, sample_intents):
        """Test report property accessors"""
        report = run_sufficiency_test(sample_intents)

        assert isinstance(report.passed, bool)
        assert report.total_criteria == report.criteria_total
        assert report.overall_score >= 0

    def test_report_to_dict(self, sample_intents):
        """Test report serialization"""
        report = run_sufficiency_test(sample_intents)
        data = report.to_dict()

        assert "result" in data
        assert "overall_score" in data
        assert "criteria" in data

    def test_report_summary_string(self, sample_intents):
        """Test summary string generation"""
        report = run_sufficiency_test(sample_intents)
        summary = report.to_summary_string()

        assert "Intent Sufficiency Test Report" in summary
        assert "Continuity" in summary


class TestSufficiencyWithAuthor:
    def test_expected_author_match(self, sample_intents):
        """Test with matching expected author"""
        report = run_sufficiency_test(sample_intents, expected_author="test_user")

        # Should not fail on author mismatch
        attribution = report.criteria.get("Human Attribution")
        if attribution:
            # No issues about unexpected author
            author_issues = [i for i in attribution.issues if "Unexpected author" in i]
            assert len(author_issues) == 0

    def test_expected_author_mismatch(self, sample_intents):
        """Test with mismatched expected author"""
        report = run_sufficiency_test(sample_intents, expected_author="different_user")

        attribution = report.criteria.get("Human Attribution")
        if attribution and attribution.issues:
            # Should have issue about unexpected author
            author_issues = [i for i in attribution.issues if "Unexpected author" in i]
            assert len(author_issues) > 0

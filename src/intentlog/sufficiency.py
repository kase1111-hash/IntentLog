"""
Intent Sufficiency Test Module

Implements the Intent Sufficiency Test from Doctrine-of-intent.md.

The test validates that intent records meet minimum requirements for
provenance verification. It checks:

1. Continuity: Duration spans a meaningful period
2. Directionality: Goals and constraints are documented
3. Resolution: Sufficient detail for the context
4. Temporal Anchoring: Timestamps are non-retroactive
5. Human Attribution: Responsible agent is identified

Returns: pass/fail with confidence score and detailed breakdown.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .core import Intent


class SufficiencyResult(Enum):
    """Result of sufficiency test"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"     # Meets some but not all criteria
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class CriterionResult:
    """Result for a single sufficiency criterion"""
    name: str
    passed: bool
    score: float                          # 0.0-1.0
    weight: float = 1.0                   # Relative importance
    details: str = ""
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "weight": self.weight,
            "details": self.details,
            "issues": self.issues,
        }


@dataclass
class SufficiencyReport:
    """
    Complete Intent Sufficiency Test report.

    Per Doctrine-of-intent.md, this validates that intent records
    can serve as evidence of genuine human effort.
    """
    result: SufficiencyResult = SufficiencyResult.INSUFFICIENT_DATA
    overall_score: float = 0.0            # 0.0-1.0
    confidence: float = 0.0               # Confidence in the result

    # Individual criteria
    continuity: Optional[CriterionResult] = None
    directionality: Optional[CriterionResult] = None
    resolution: Optional[CriterionResult] = None
    temporal_anchoring: Optional[CriterionResult] = None
    human_attribution: Optional[CriterionResult] = None

    # Summary
    criteria_passed: int = 0
    criteria_total: int = 5
    recommendations: List[str] = field(default_factory=list)

    @property
    def total_criteria(self) -> int:
        """Alias for criteria_total"""
        return self.criteria_total

    @property
    def passed(self) -> bool:
        """True if test passed"""
        return self.result == SufficiencyResult.PASS

    @property
    def criteria(self) -> Dict[str, CriterionResult]:
        """Return criteria as dictionary"""
        result = {}
        if self.continuity:
            result["Continuity"] = self.continuity
        if self.directionality:
            result["Directionality"] = self.directionality
        if self.resolution:
            result["Resolution"] = self.resolution
        if self.temporal_anchoring:
            result["Temporal Anchoring"] = self.temporal_anchoring
        if self.human_attribution:
            result["Human Attribution"] = self.human_attribution
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.value,
            "overall_score": round(self.overall_score, 4),
            "confidence": round(self.confidence, 4),
            "criteria": {
                "continuity": self.continuity.to_dict() if self.continuity else None,
                "directionality": self.directionality.to_dict() if self.directionality else None,
                "resolution": self.resolution.to_dict() if self.resolution else None,
                "temporal_anchoring": self.temporal_anchoring.to_dict() if self.temporal_anchoring else None,
                "human_attribution": self.human_attribution.to_dict() if self.human_attribution else None,
            },
            "summary": {
                "criteria_passed": self.criteria_passed,
                "criteria_total": self.criteria_total,
                "pass_rate": round(self.criteria_passed / self.criteria_total, 4) if self.criteria_total > 0 else 0,
            },
            "recommendations": self.recommendations,
        }

    def to_summary_string(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 50,
            "Intent Sufficiency Test Report",
            "=" * 50,
            "",
            f"Result: {self.result.value.upper()}",
            f"Overall Score: {self.overall_score:.2%}",
            f"Confidence: {self.confidence:.2%}",
            "",
            f"Criteria Passed: {self.criteria_passed}/{self.criteria_total}",
            "",
        ]

        # Individual criteria
        criteria = [
            ("Continuity", self.continuity),
            ("Directionality", self.directionality),
            ("Resolution", self.resolution),
            ("Temporal Anchoring", self.temporal_anchoring),
            ("Human Attribution", self.human_attribution),
        ]

        for name, crit in criteria:
            if crit:
                status = "PASS" if crit.passed else "FAIL"
                lines.append(f"  {name}: {status} ({crit.score:.2%})")
                if crit.issues:
                    for issue in crit.issues:
                        lines.append(f"    - {issue}")

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


class SufficiencyTest:
    """
    Intent Sufficiency Test implementation.

    Tests whether intent records meet the minimum requirements
    for provenance verification as defined in Doctrine-of-intent.md.
    """

    # Thresholds for each criterion
    MIN_DURATION_HOURS = 0.5              # Minimum time span
    MIN_INTENTS = 3                       # Minimum number of intents
    MIN_REASONING_WORDS = 5               # Minimum words in reasoning
    MAX_TIMESTAMP_DRIFT_SECONDS = 60      # Max allowed backward drift
    GOAL_KEYWORDS = [                     # Keywords indicating goals
        "goal", "objective", "aim", "target", "want", "need",
        "should", "must", "will", "plan", "intend", "decide",
    ]
    CONSTRAINT_KEYWORDS = [               # Keywords indicating constraints
        "constraint", "limit", "cannot", "must not", "avoid",
        "requirement", "boundary", "restriction", "should not",
    ]

    def __init__(
        self,
        intents: List[Intent],
        expected_author: Optional[str] = None,
    ):
        """
        Initialize sufficiency test.

        Args:
            intents: List of intents to test
            expected_author: Expected author/agent identifier
        """
        self.intents = sorted(intents, key=lambda i: i.timestamp)
        self.expected_author = expected_author

    def run(self) -> SufficiencyReport:
        """
        Run the complete sufficiency test.

        Returns comprehensive report with pass/fail and confidence.
        """
        if len(self.intents) < self.MIN_INTENTS:
            return SufficiencyReport(
                result=SufficiencyResult.INSUFFICIENT_DATA,
                recommendations=[
                    f"Need at least {self.MIN_INTENTS} intents for sufficiency test",
                    f"Current count: {len(self.intents)}",
                ],
            )

        # Run individual tests
        continuity = self._test_continuity()
        directionality = self._test_directionality()
        resolution = self._test_resolution()
        temporal = self._test_temporal_anchoring()
        attribution = self._test_human_attribution()

        criteria = [continuity, directionality, resolution, temporal, attribution]

        # Calculate overall score (weighted average)
        total_weight = sum(c.weight for c in criteria)
        weighted_score = sum(c.score * c.weight for c in criteria) / total_weight

        # Count passed
        passed_count = sum(1 for c in criteria if c.passed)

        # Determine result
        if passed_count == 5:
            result = SufficiencyResult.PASS
        elif passed_count >= 3:
            result = SufficiencyResult.PARTIAL
        else:
            result = SufficiencyResult.FAIL

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(criteria)

        # Generate recommendations
        recommendations = self._generate_recommendations(criteria)

        return SufficiencyReport(
            result=result,
            overall_score=weighted_score,
            confidence=confidence,
            continuity=continuity,
            directionality=directionality,
            resolution=resolution,
            temporal_anchoring=temporal,
            human_attribution=attribution,
            criteria_passed=passed_count,
            recommendations=recommendations,
        )

    def _test_continuity(self) -> CriterionResult:
        """
        Test 1: Continuity

        Validates that the duration spans a meaningful period
        and shows continuous effort over time.
        """
        issues = []

        if len(self.intents) < 2:
            return CriterionResult(
                name="continuity",
                passed=False,
                score=0.0,
                weight=1.0,
                details="Insufficient intents for continuity analysis",
                issues=["Need at least 2 intents"],
            )

        # Calculate duration
        first_time = self.intents[0].timestamp
        last_time = self.intents[-1].timestamp
        duration_hours = (last_time - first_time).total_seconds() / 3600

        if duration_hours < self.MIN_DURATION_HOURS:
            issues.append(f"Duration too short: {duration_hours:.2f} hours")
            duration_score = duration_hours / self.MIN_DURATION_HOURS
        else:
            duration_score = min(1.0, duration_hours / (self.MIN_DURATION_HOURS * 4))

        # Check for large gaps
        gaps = []
        for i in range(1, len(self.intents)):
            gap = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds() / 60
            gaps.append(gap)

        max_gap = max(gaps) if gaps else 0
        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        # Penalize very large gaps
        gap_penalty = 0
        if max_gap > 120:  # 2 hour gap
            gap_penalty = min(0.3, (max_gap - 120) / 360)
            issues.append(f"Large gap detected: {max_gap:.0f} minutes")

        # Calculate continuity score
        score = max(0, duration_score - gap_penalty)
        passed = score >= 0.5

        return CriterionResult(
            name="continuity",
            passed=passed,
            score=score,
            weight=1.0,
            details=f"Duration: {duration_hours:.2f}h, Max gap: {max_gap:.0f}min",
            issues=issues,
        )

    def _test_directionality(self) -> CriterionResult:
        """
        Test 2: Directionality

        Validates that goals and constraints are documented.
        """
        issues = []

        # Search for goal indicators
        goal_count = 0
        constraint_count = 0

        for intent in self.intents:
            text = (intent.intent_name + " " + intent.intent_reasoning).lower()

            for keyword in self.GOAL_KEYWORDS:
                if keyword in text:
                    goal_count += 1
                    break

            for keyword in self.CONSTRAINT_KEYWORDS:
                if keyword in text:
                    constraint_count += 1
                    break

        # Score based on presence of directional language
        total = len(self.intents)
        goal_ratio = goal_count / total if total > 0 else 0
        constraint_ratio = constraint_count / total if total > 0 else 0

        # At least some goals should be documented
        if goal_ratio < 0.1:
            issues.append("Few intents document explicit goals")

        # Constraints are less common but should exist
        if constraint_ratio == 0 and len(self.intents) > 10:
            issues.append("No constraints or limitations documented")

        # Calculate score
        score = min(1.0, goal_ratio * 2 + constraint_ratio)
        passed = goal_ratio >= 0.1

        return CriterionResult(
            name="directionality",
            passed=passed,
            score=score,
            weight=1.0,
            details=f"Goals mentioned: {goal_count}/{total}, Constraints: {constraint_count}",
            issues=issues,
        )

    def _test_resolution(self) -> CriterionResult:
        """
        Test 3: Resolution

        Validates sufficient detail for the context.
        """
        issues = []

        # Check reasoning length
        word_counts = [len(i.intent_reasoning.split()) for i in self.intents]
        avg_words = sum(word_counts) / len(word_counts)

        short_count = sum(1 for w in word_counts if w < self.MIN_REASONING_WORDS)
        short_ratio = short_count / len(word_counts)

        if short_ratio > 0.5:
            issues.append(f"{short_count} intents have very brief reasoning")

        if avg_words < self.MIN_REASONING_WORDS:
            issues.append(f"Average reasoning too short: {avg_words:.1f} words")

        # Check for metadata presence (indicates detail)
        metadata_counts = [len(i.metadata) for i in self.intents]
        avg_metadata = sum(metadata_counts) / len(metadata_counts)

        # Calculate score
        word_score = min(1.0, avg_words / 20)  # Target 20 words
        metadata_score = min(1.0, avg_metadata / 2)  # Target 2 metadata keys

        score = word_score * 0.7 + metadata_score * 0.3
        passed = score >= 0.5

        return CriterionResult(
            name="resolution",
            passed=passed,
            score=score,
            weight=1.0,
            details=f"Avg words: {avg_words:.1f}, Avg metadata: {avg_metadata:.1f}",
            issues=issues,
        )

    def _test_temporal_anchoring(self) -> CriterionResult:
        """
        Test 4: Temporal Anchoring

        Validates that timestamps are non-retroactive and plausible.
        """
        issues = []

        # Check for backward timestamps
        backward_count = 0
        large_backward_count = 0

        for i in range(1, len(self.intents)):
            diff = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds()
            if diff < 0:
                backward_count += 1
                if abs(diff) > self.MAX_TIMESTAMP_DRIFT_SECONDS:
                    large_backward_count += 1

        if backward_count > 0:
            issues.append(f"{backward_count} timestamps are out of order")

        if large_backward_count > 0:
            issues.append(f"{large_backward_count} timestamps have significant retroactive drift")

        # Check for future timestamps
        now = datetime.now()
        future_count = sum(1 for i in self.intents if i.timestamp > now + timedelta(minutes=5))
        if future_count > 0:
            issues.append(f"{future_count} timestamps are in the future")

        # Calculate score
        total = len(self.intents)
        backward_ratio = backward_count / total if total > 0 else 0
        future_ratio = future_count / total if total > 0 else 0

        score = max(0, 1.0 - backward_ratio * 2 - future_ratio * 3)
        passed = score >= 0.8

        return CriterionResult(
            name="temporal_anchoring",
            passed=passed,
            score=score,
            weight=1.2,  # Slightly higher weight - important for verification
            details=f"Backward: {backward_count}, Future: {future_count}",
            issues=issues,
        )

    def _test_human_attribution(self) -> CriterionResult:
        """
        Test 5: Human Attribution

        Validates that a responsible agent is identified.
        """
        issues = []

        # Check for author/agent in metadata
        attributed_count = 0

        for intent in self.intents:
            author = intent.metadata.get("author") or intent.metadata.get("agent")
            user = intent.metadata.get("user") or intent.metadata.get("user_id")

            if author or user:
                attributed_count += 1

                # Check against expected author if provided
                if self.expected_author:
                    actual = author or user
                    if actual != self.expected_author:
                        issues.append(f"Unexpected author: {actual}")

        # Calculate score
        attribution_ratio = attributed_count / len(self.intents) if self.intents else 0

        if attribution_ratio < 0.5:
            issues.append("Many intents lack author/agent attribution")

        # Check for session or context identifiers
        session_present = any(
            "session" in k.lower() or "context" in k.lower()
            for i in self.intents
            for k in i.metadata.keys()
        )

        score = attribution_ratio * 0.8
        if session_present:
            score += 0.2  # Bonus for session tracking

        passed = score >= 0.5 or attribution_ratio >= 0.3

        return CriterionResult(
            name="human_attribution",
            passed=passed,
            score=min(1.0, score),
            weight=1.0,
            details=f"Attributed: {attributed_count}/{len(self.intents)}",
            issues=issues,
        )

    def _calculate_confidence(self, criteria: List[CriterionResult]) -> float:
        """Calculate confidence in the test result"""
        # Confidence based on:
        # - Data volume (more intents = higher confidence)
        # - Criterion agreement (all pass/fail = higher confidence)
        # - Score variance (low variance = higher confidence)

        # Volume factor
        n = len(self.intents)
        if n >= 50:
            volume_factor = 1.0
        elif n >= 20:
            volume_factor = 0.8
        elif n >= 10:
            volume_factor = 0.6
        else:
            volume_factor = 0.4

        # Agreement factor
        passed_count = sum(1 for c in criteria if c.passed)
        if passed_count == 5 or passed_count == 0:
            agreement_factor = 1.0
        elif passed_count >= 4 or passed_count <= 1:
            agreement_factor = 0.8
        else:
            agreement_factor = 0.6

        # Variance factor
        scores = [c.score for c in criteria]
        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            variance_factor = max(0.5, 1.0 - variance)
        else:
            variance_factor = 0.5

        return (volume_factor * 0.4 + agreement_factor * 0.3 + variance_factor * 0.3)

    def _generate_recommendations(self, criteria: List[CriterionResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        for crit in criteria:
            if not crit.passed:
                if crit.name == "continuity":
                    recommendations.append(
                        "Increase logging frequency and reduce gaps between entries"
                    )
                elif crit.name == "directionality":
                    recommendations.append(
                        "Document explicit goals and constraints in intent reasoning"
                    )
                elif crit.name == "resolution":
                    recommendations.append(
                        "Provide more detailed reasoning and include relevant metadata"
                    )
                elif crit.name == "temporal_anchoring":
                    recommendations.append(
                        "Ensure timestamps are generated at intent creation time"
                    )
                elif crit.name == "human_attribution":
                    recommendations.append(
                        "Add author/agent identifier to intent metadata"
                    )

        return recommendations


def run_sufficiency_test(
    intents: List[Intent],
    expected_author: Optional[str] = None,
) -> SufficiencyReport:
    """
    Convenience function to run sufficiency test.

    Args:
        intents: List of intents to test
        expected_author: Optional expected author identifier

    Returns:
        SufficiencyReport with pass/fail and detailed breakdown
    """
    test = SufficiencyTest(intents, expected_author)
    return test.run()


# Keep convenience function available (note: not named test_* to avoid pytest collision)
# Use run_sufficiency_test() or SufficiencyTest(intents).run()
sufficiency_test = run_sufficiency_test  # Alias

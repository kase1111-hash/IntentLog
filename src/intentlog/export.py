"""
Export Module for IntentLog

Provides functionality to export intents for:
- Evaluation set generation (ground truth for testing)
- Fine-tuning data pipelines
- Analytics and reporting

Supports filtering, anonymization, and multiple output formats.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import hashlib
import json
import re
import uuid

from .core import Intent


# ReDoS protection constants
MAX_REGEX_INPUT_LENGTH = 10000  # Max chars to match against
REGEX_COMPILE_CACHE: Dict[str, Any] = {}


def safe_regex_match(pattern: str, text: str, flags: int = 0) -> bool:
    """
    Safely match a regex pattern against text with ReDoS protection.

    Args:
        pattern: Regex pattern to match
        text: Text to match against
        flags: Regex flags (e.g., re.IGNORECASE)

    Returns:
        True if pattern matches, False otherwise

    Raises:
        ValueError: If pattern is invalid
    """
    # Limit input length to prevent catastrophic backtracking
    if len(text) > MAX_REGEX_INPUT_LENGTH:
        text = text[:MAX_REGEX_INPUT_LENGTH]

    # Cache compiled patterns for performance
    cache_key = f"{pattern}:{flags}"
    if cache_key not in REGEX_COMPILE_CACHE:
        try:
            REGEX_COMPILE_CACHE[cache_key] = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    compiled = REGEX_COMPILE_CACHE[cache_key]
    return compiled.search(text) is not None


@dataclass
class ExportFilter:
    """
    Filters for selecting intents to export.

    All filters are optional - only specified filters are applied.
    """
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_latency_ms: Optional[int] = None    # Minimum latency threshold
    max_latency_ms: Optional[int] = None    # Maximum latency threshold
    categories: Optional[List[str]] = None  # Filter by metadata category
    tags: Optional[List[str]] = None        # Filter by metadata tags
    name_pattern: Optional[str] = None      # Regex pattern for intent name
    reasoning_pattern: Optional[str] = None # Regex pattern for reasoning
    has_parent: Optional[bool] = None       # Filter by parent presence
    has_metadata_key: Optional[str] = None  # Must have specific metadata key
    custom_filter: Optional[Callable[[Intent], bool]] = None

    def matches(self, intent: Intent) -> bool:
        """Check if intent matches all specified filters"""
        # Date filters
        if self.start_date and intent.timestamp < self.start_date:
            return False
        if self.end_date and intent.timestamp > self.end_date:
            return False

        # Latency filters (from metadata)
        latency = intent.metadata.get("latency_ms")
        if latency is not None:
            if self.min_latency_ms and latency < self.min_latency_ms:
                return False
            if self.max_latency_ms and latency > self.max_latency_ms:
                return False
        elif self.min_latency_ms or self.max_latency_ms:
            # Latency filter specified but intent has no latency
            return False

        # Category filter
        if self.categories:
            intent_category = intent.metadata.get("category", "")
            if intent_category not in self.categories:
                return False

        # Tags filter
        if self.tags:
            intent_tags = intent.metadata.get("tags", [])
            if not any(tag in intent_tags for tag in self.tags):
                return False

        # Name pattern (with ReDoS protection)
        if self.name_pattern:
            try:
                if not safe_regex_match(self.name_pattern, intent.intent_name, re.IGNORECASE):
                    return False
            except ValueError:
                return False  # Invalid pattern never matches

        # Reasoning pattern (with ReDoS protection)
        if self.reasoning_pattern:
            try:
                if not safe_regex_match(self.reasoning_pattern, intent.intent_reasoning, re.IGNORECASE):
                    return False
            except ValueError:
                return False  # Invalid pattern never matches

        # Parent filter
        if self.has_parent is not None:
            has_parent = intent.parent_intent_id is not None
            if has_parent != self.has_parent:
                return False

        # Metadata key filter
        if self.has_metadata_key:
            if self.has_metadata_key not in intent.metadata:
                return False

        # Custom filter
        if self.custom_filter:
            if not self.custom_filter(intent):
                return False

        return True


@dataclass
class AnonymizationConfig:
    """Configuration for anonymizing exported data"""
    anonymize_names: bool = True      # Replace intent names with generic labels
    anonymize_reasoning: bool = False # Anonymize reasoning text (lossy)
    hash_ids: bool = True             # Hash intent IDs
    remove_timestamps: bool = False   # Remove exact timestamps
    round_timestamps: str = ""        # Round to: "hour", "day", "week"
    remove_metadata_keys: List[str] = field(default_factory=list)
    keep_only_metadata_keys: List[str] = field(default_factory=list)
    salt: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def anonymize_intent(self, intent: Intent, index: int = 0) -> Dict[str, Any]:
        """Anonymize a single intent"""
        data = {}

        # ID
        if self.hash_ids:
            data["intent_id"] = hashlib.sha256(
                f"{self.salt}:{intent.intent_id}".encode()
            ).hexdigest()[:16]
        else:
            data["intent_id"] = intent.intent_id

        # Name
        if self.anonymize_names:
            data["intent_name"] = f"Intent_{index:04d}"
        else:
            data["intent_name"] = intent.intent_name

        # Reasoning
        if self.anonymize_reasoning:
            # Replace with length indicator
            word_count = len(intent.intent_reasoning.split())
            data["intent_reasoning"] = f"[Reasoning: {word_count} words]"
        else:
            data["intent_reasoning"] = intent.intent_reasoning

        # Timestamp
        if self.remove_timestamps:
            data["timestamp"] = None
        elif self.round_timestamps:
            ts = intent.timestamp
            if self.round_timestamps == "hour":
                ts = ts.replace(minute=0, second=0, microsecond=0)
            elif self.round_timestamps == "day":
                ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            elif self.round_timestamps == "week":
                # Round to start of week
                ts = ts - timedelta(days=ts.weekday())
                ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            data["timestamp"] = ts.isoformat()
        else:
            data["timestamp"] = intent.timestamp.isoformat()

        # Parent ID
        if intent.parent_intent_id:
            if self.hash_ids:
                data["parent_id"] = hashlib.sha256(
                    f"{self.salt}:{intent.parent_intent_id}".encode()
                ).hexdigest()[:16]
            else:
                data["parent_id"] = intent.parent_intent_id
        else:
            data["parent_id"] = None

        # Metadata
        metadata = intent.metadata.copy()
        if self.remove_metadata_keys:
            for key in self.remove_metadata_keys:
                metadata.pop(key, None)
        if self.keep_only_metadata_keys:
            metadata = {k: v for k, v in metadata.items()
                       if k in self.keep_only_metadata_keys}
        data["metadata"] = metadata

        return data


@dataclass
class ExportFormat:
    """Output format configuration"""
    format_type: str = "jsonl"  # "json", "jsonl", "csv", "huggingface", "openai"
    include_metadata: bool = True
    include_parent: bool = True
    include_timestamp: bool = True
    pretty_print: bool = False

    # For eval sets
    include_expected_outcome: bool = False
    outcome_field: str = "expected_outcome"

    # For fine-tuning
    system_prompt: str = ""
    user_template: str = "{intent_name}"
    assistant_template: str = "{intent_reasoning}"


class IntentExporter:
    """
    Exports intents to various formats for evaluation and training.
    """

    def __init__(
        self,
        filter_config: Optional[ExportFilter] = None,
        anonymization: Optional[AnonymizationConfig] = None,
        format_config: Optional[ExportFormat] = None,
    ):
        self.filter = filter_config or ExportFilter()
        self.anonymization = anonymization
        self.format = format_config or ExportFormat()

    def export(
        self,
        intents: List[Intent],
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Export intents according to configuration.

        Args:
            intents: List of intents to export
            output_path: Optional path to write output (returns string if None)

        Returns:
            Exported data as string
        """
        # Filter intents
        filtered = [i for i in intents if self.filter.matches(i)]

        # Convert to dicts (with optional anonymization)
        if self.anonymization:
            data = [
                self.anonymization.anonymize_intent(intent, idx)
                for idx, intent in enumerate(filtered)
            ]
        else:
            data = [self._intent_to_dict(intent) for intent in filtered]

        # Format output
        if self.format.format_type == "json":
            output = self._format_json(data)
        elif self.format.format_type == "jsonl":
            output = self._format_jsonl(data)
        elif self.format.format_type == "csv":
            output = self._format_csv(data)
        elif self.format.format_type == "huggingface":
            output = self._format_huggingface(data)
        elif self.format.format_type == "openai":
            output = self._format_openai(data)
        else:
            output = self._format_jsonl(data)

        # Write to file if path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)

        return output

    def _intent_to_dict(self, intent: Intent) -> Dict[str, Any]:
        """Convert intent to dictionary"""
        data = {
            "intent_id": intent.intent_id,
            "intent_name": intent.intent_name,
            "intent_reasoning": intent.intent_reasoning,
        }

        if self.format.include_timestamp:
            data["timestamp"] = intent.timestamp.isoformat()

        if self.format.include_parent and intent.parent_intent_id:
            data["parent_id"] = intent.parent_intent_id

        if self.format.include_metadata and intent.metadata:
            data["metadata"] = intent.metadata

        return data

    def _format_json(self, data: List[Dict]) -> str:
        """Format as JSON array"""
        if self.format.pretty_print:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    def _format_jsonl(self, data: List[Dict]) -> str:
        """Format as JSON Lines"""
        lines = [json.dumps(item, default=str) for item in data]
        return "\n".join(lines)

    def _format_csv(self, data: List[Dict]) -> str:
        """Format as CSV"""
        if not data:
            return ""

        # Get all unique keys
        keys = []
        for item in data:
            for key in item.keys():
                if key not in keys and key != "metadata":
                    keys.append(key)

        lines = [",".join(keys)]
        for item in data:
            values = []
            for key in keys:
                val = item.get(key, "")
                # Escape CSV values
                if isinstance(val, str):
                    val = val.replace('"', '""')
                    if "," in val or "\n" in val or '"' in val:
                        val = f'"{val}"'
                else:
                    val = str(val) if val is not None else ""
                values.append(val)
            lines.append(",".join(values))

        return "\n".join(lines)

    def _format_huggingface(self, data: List[Dict]) -> str:
        """Format for HuggingFace datasets"""
        hf_data = []
        for item in data:
            hf_item = {
                "text": self.format.user_template.format(**item),
                "label": item.get("intent_name", ""),
            }
            if self.format.include_expected_outcome:
                hf_item["expected"] = item.get(self.format.outcome_field, "")
            hf_data.append(hf_item)

        return self._format_jsonl(hf_data)

    def _format_openai(self, data: List[Dict]) -> str:
        """Format for OpenAI fine-tuning"""
        openai_data = []
        for item in data:
            messages = []

            if self.format.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.format.system_prompt,
                })

            messages.append({
                "role": "user",
                "content": self.format.user_template.format(**item),
            })

            messages.append({
                "role": "assistant",
                "content": self.format.assistant_template.format(**item),
            })

            openai_data.append({"messages": messages})

        return self._format_jsonl(openai_data)

    def get_stats(self, intents: List[Intent]) -> Dict[str, Any]:
        """Get export statistics"""
        filtered = [i for i in intents if self.filter.matches(i)]

        return {
            "total_intents": len(intents),
            "filtered_intents": len(filtered),
            "filter_ratio": len(filtered) / len(intents) if intents else 0,
            "date_range": {
                "start": min(i.timestamp for i in filtered).isoformat() if filtered else None,
                "end": max(i.timestamp for i in filtered).isoformat() if filtered else None,
            },
            "anonymization_enabled": self.anonymization is not None,
            "format": self.format.format_type,
        }


def export_for_eval(
    intents: List[Intent],
    output_path: Path,
    include_outcomes: bool = True,
    anonymize: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to export intents as an evaluation set.

    Args:
        intents: List of intents
        output_path: Path to write eval set
        include_outcomes: Whether to include expected outcomes
        anonymize: Whether to anonymize data

    Returns:
        Export statistics
    """
    anonymization = AnonymizationConfig() if anonymize else None
    format_config = ExportFormat(
        format_type="jsonl",
        include_expected_outcome=include_outcomes,
        pretty_print=False,
    )

    exporter = IntentExporter(
        anonymization=anonymization,
        format_config=format_config,
    )

    exporter.export(intents, output_path)
    return exporter.get_stats(intents)


def export_for_finetuning(
    intents: List[Intent],
    output_path: Path,
    format_type: str = "openai",
    system_prompt: str = "You are analyzing intent and reasoning patterns.",
    filter_config: Optional[ExportFilter] = None,
) -> Dict[str, Any]:
    """
    Convenience function to export intents for fine-tuning.

    Args:
        intents: List of intents
        output_path: Path to write training data
        format_type: "openai" or "huggingface"
        system_prompt: System prompt for fine-tuning
        filter_config: Optional filters to apply

    Returns:
        Export statistics
    """
    format_config = ExportFormat(
        format_type=format_type,
        system_prompt=system_prompt,
        user_template="Intent: {intent_name}\nContext: {intent_reasoning}",
        assistant_template="{intent_reasoning}",
    )

    exporter = IntentExporter(
        filter_config=filter_config,
        format_config=format_config,
    )

    exporter.export(intents, output_path)
    return exporter.get_stats(intents)

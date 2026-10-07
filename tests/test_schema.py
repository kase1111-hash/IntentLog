"""
Tests for schema validation and the project health check.

These tests run against real projects on disk and need no optional
dependencies (intents are chained but never signed).
"""

import json
from pathlib import Path

import pytest

from intentlog.schema import (
    SchemaValidator,
    check_project_health,
    validate_config_file,
    validate_intents_file,
    validate_project,
)
from intentlog.storage import CONFIG_FILE, INTENTLOG_DIR, INTENTS_FILE, IntentLogStorage


@pytest.fixture
def project(tmp_path):
    """A freshly initialized, empty project."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    IntentLogStorage(project_dir).init_project("health-test")
    return project_dir


def _intents_file(project_dir: Path) -> Path:
    return project_dir / INTENTLOG_DIR / INTENTS_FILE


def _add_chained(project_dir: Path, count: int) -> None:
    storage = IntentLogStorage(project_dir)
    for i in range(count):
        storage.add_chained_intent(f"intent {i}", f"reasoning for intent {i}")


def _rewrite_intents(project_dir: Path, mutate) -> None:
    """Load intents.json, apply ``mutate(data)`` and write it back."""
    path = _intents_file(project_dir)
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2))


# ---- check_project_health ----


class TestCheckProjectHealth:
    def test_empty_project_is_healthy(self, project):
        result = check_project_health(project)

        assert result["healthy"] is True
        assert result["project_found"] is True
        assert result["issues"] == []
        assert result["checks"]["schema_validation"] == {"passed": True, "issues": []}
        assert result["checks"]["chain_integrity"] == {
            "passed": True,
            "issues": [],
            "warnings": [],
        }
        assert result["checks"]["intent_count"] == {"passed": True, "count": 0}

    def test_project_with_chained_intents_is_healthy(self, project):
        _add_chained(project, 3)

        result = check_project_health(project)

        assert result["healthy"] is True
        assert result["issues"] == []
        assert result["checks"]["chain_integrity"]["passed"] is True
        assert result["checks"]["chain_integrity"]["issues"] == []
        assert result["checks"]["intent_count"]["count"] == 3

    def test_project_with_plain_intents_is_healthy(self, project):
        # Legacy intents without chain data are chained on load.
        storage = IntentLogStorage(project)
        storage.add_intent("plain one", "no chain data on disk")
        storage.add_intent("plain two", "still no chain data on disk")

        result = check_project_health(project)

        assert result["healthy"] is True
        assert result["checks"]["chain_integrity"]["passed"] is True
        assert result["checks"]["intent_count"]["count"] == 2

    def test_tampered_intent_content_is_reported(self, project):
        _add_chained(project, 2)

        def tamper(data):
            data["intents"][0]["intent_reasoning"] = "rewritten after the fact"

        _rewrite_intents(project, tamper)

        result = check_project_health(project)

        chain = result["checks"]["chain_integrity"]
        assert result["healthy"] is False
        assert chain["passed"] is False
        assert chain["issues"]
        assert any("hash mismatch" in issue.lower() for issue in chain["issues"])
        assert any(issue.startswith("Chain integrity: ") for issue in result["issues"])
        assert not any("has no attribute" in issue for issue in result["issues"])

    def test_removed_link_is_reported_as_chain_break(self, project):
        _add_chained(project, 3)

        def drop_middle(data):
            data["intents"].pop(1)

        _rewrite_intents(project, drop_middle)

        result = check_project_health(project)

        assert result["healthy"] is False
        assert result["checks"]["chain_integrity"]["passed"] is False
        assert any("chain break" in issue.lower() for issue in result["issues"])

    def test_corrupted_intents_file_is_reported(self, project):
        _add_chained(project, 1)
        _intents_file(project).write_text("{not valid json")

        result = check_project_health(project)

        assert result["healthy"] is False
        assert result["checks"]["schema_validation"]["passed"] is False
        assert any("Invalid JSON" in issue for issue in result["issues"])
        assert result["checks"]["chain_integrity"]["passed"] is False
        assert result["checks"]["chain_integrity"]["issues"]
        assert result["checks"]["intent_count"]["passed"] is False

    def test_directory_without_project_is_unhealthy(self, tmp_path):
        result = check_project_health(tmp_path / "nothing-here")

        assert result["healthy"] is False
        assert "No .intentlog directory found" in result["issues"]
        assert result["checks"]["chain_integrity"]["passed"] is False


# ---- Schema validation ----


class TestSchemaValidation:
    def test_validate_project_after_init(self, project):
        assert validate_project(project) == (True, [])

    def test_chained_intents_file_matches_schema(self, project):
        _add_chained(project, 2)
        assert validate_intents_file(_intents_file(project)).valid

    def test_config_with_invalid_project_name_fails(self, tmp_path):
        config_path = tmp_path / CONFIG_FILE
        config_path.write_text(json.dumps({"project_name": "bad name!"}))

        result = validate_config_file(config_path)

        assert result.valid is False
        assert any("project_name" in message for message in result.error_messages)

    def test_intents_file_with_unknown_top_level_key_fails(self, tmp_path):
        path = tmp_path / INTENTS_FILE
        path.write_text(json.dumps({"branch": "main", "intents": [], "bogus": 1}))

        result = validate_intents_file(path)

        assert result.valid is False
        assert any("bogus" in message for message in result.error_messages)

    def test_unknown_schema_name(self):
        result = SchemaValidator().validate({}, "nope")
        assert result.valid is False
        assert "Unknown schema" in result.error_messages[0]

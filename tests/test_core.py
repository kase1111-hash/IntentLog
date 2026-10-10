"""Tests for IntentLog core functionality"""
import pytest
from intentlog.core import Intent, IntentLog


def test_intent_creation():
    """Test creating a basic intent"""
    intent = Intent(
        intent_name="test_action",
        intent_reasoning="This is a test"
    )

    assert intent.intent_name == "test_action"
    assert intent.intent_reasoning == "This is a test"
    assert intent.intent_id  # Should have auto-generated ID


def test_intent_validation():
    """Test intent validation"""
    # Valid intent
    valid_intent = Intent(
        intent_name="valid",
        intent_reasoning="Proper reasoning"
    )
    assert valid_intent.validate() is True

    # Invalid - empty name
    invalid_intent1 = Intent(
        intent_name="",
        intent_reasoning="Has reasoning"
    )
    assert invalid_intent1.validate() is False

    # Invalid - empty reasoning
    invalid_intent2 = Intent(
        intent_name="has_name",
        intent_reasoning=""
    )
    assert invalid_intent2.validate() is False


def test_intent_to_dict():
    """Test converting intent to dictionary"""
    intent = Intent(
        intent_name="export_test",
        intent_reasoning="Testing export",
        metadata={"key": "value"}
    )

    intent_dict = intent.to_dict()

    assert intent_dict["intent_name"] == "export_test"
    assert intent_dict["intent_reasoning"] == "Testing export"
    assert intent_dict["metadata"]["key"] == "value"
    assert "timestamp" in intent_dict
    assert "intent_id" in intent_dict


def test_intentlog_initialization():
    """Test IntentLog initialization"""
    log = IntentLog(project_name="test_project")

    assert log.project_name == "test_project"
    assert log.current_branch == "main"
    assert len(log.intents) == 0


def test_add_intent():
    """Test adding intents to log"""
    log = IntentLog()

    intent = log.add_intent(
        name="first_action",
        reasoning="First test action"
    )

    assert len(log.intents) == 1
    assert intent.intent_name == "first_action"
    assert intent.intent_reasoning == "First test action"


def test_add_intent_with_parent():
    """Test adding nested intents with parent relationship"""
    log = IntentLog()

    parent = log.add_intent(
        name="parent_action",
        reasoning="Parent reasoning"
    )

    child = log.add_intent(
        name="child_action",
        reasoning="Child reasoning",
        parent_id=parent.intent_id
    )

    assert child.parent_intent_id == parent.intent_id


def test_add_invalid_intent():
    """Test that adding invalid intent raises error"""
    log = IntentLog()

    with pytest.raises(ValueError):
        log.add_intent(name="test", reasoning="")


def test_get_intent():
    """Test retrieving intent by ID"""
    log = IntentLog()

    intent = log.add_intent(name="findme", reasoning="Test")
    retrieved = log.get_intent(intent.intent_id)

    assert retrieved is not None
    assert retrieved.intent_id == intent.intent_id


def test_get_intent_chain():
    """Test retrieving full intent chain"""
    log = IntentLog()

    root = log.add_intent(name="root", reasoning="Root action")
    child1 = log.add_intent(name="child1", reasoning="Child 1", parent_id=root.intent_id)
    child2 = log.add_intent(name="child2", reasoning="Child 2", parent_id=child1.intent_id)

    chain = log.get_intent_chain(child2.intent_id)

    assert len(chain) == 3
    assert chain[0].intent_id == root.intent_id
    assert chain[1].intent_id == child1.intent_id
    assert chain[2].intent_id == child2.intent_id


def test_search_intents():
    """Test searching intents"""
    log = IntentLog()

    log.add_intent(name="setup", reasoning="Initialize the system")
    log.add_intent(name="process", reasoning="Process the data")
    log.add_intent(name="cleanup", reasoning="Clean up resources")

    # Search by name
    results = log.search_intents("setup")
    assert len(results) == 1
    assert results[0].intent_name == "setup"

    # Search by reasoning
    results = log.search_intents("data")
    assert len(results) == 1
    assert results[0].intent_name == "process"


def test_export_to_dict():
    """Test exporting entire log to dictionary"""
    log = IntentLog(project_name="export_test")
    log.add_intent(name="action1", reasoning="First action")
    log.add_intent(name="action2", reasoning="Second action")

    export = log.export_to_dict()

    assert export["project_name"] == "export_test"
    assert export["current_branch"] == "main"
    assert len(export["intents"]) == 2

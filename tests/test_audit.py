"""Tests for IntentLog audit functionality"""
import pytest
import tempfile
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from audit_intents import audit_logs


def test_audit_passes_with_valid_logs():
    """Test that audit passes with properly formatted intent logs"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write("intent_name: 'setup'\n")
        f.write("intent_reasoning: 'Initialize the system'\n")
        f.write("intent_name: 'process'\n")
        f.write("intent_reasoning: 'Process the data'\n")
        temp_path = f.name

    try:
        # Should exit with 0 (success)
        audit_logs(temp_path)
    except SystemExit as e:
        assert e.code == 0, "Audit should pass for valid logs"
    finally:
        os.unlink(temp_path)


def test_audit_fails_with_empty_reasoning():
    """Test that audit fails when empty reasoning is found"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write("intent_name: 'test'\n")
        f.write("intent_reasoning: ''\n")  # Empty reasoning
        temp_path = f.name

    try:
        with pytest.raises(SystemExit) as exc_info:
            audit_logs(temp_path)
        assert exc_info.value.code == 1, "Audit should fail for empty reasoning"
    finally:
        os.unlink(temp_path)


def test_audit_detects_loops():
    """Test that audit detects when same intent is repeated too many times"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        # Repeat same intent 5 times (threshold is 3)
        for i in range(5):
            f.write("intent_name: 'repeated_action'\n")
            f.write("intent_reasoning: 'Doing something'\n")
        temp_path = f.name

    try:
        with pytest.raises(SystemExit) as exc_info:
            audit_logs(temp_path)
        assert exc_info.value.code == 1, "Audit should fail for repeated intents"
    finally:
        os.unlink(temp_path)


def test_basic_functionality():
    """Basic smoke test to ensure tests run"""
    assert True, "Basic test passes"

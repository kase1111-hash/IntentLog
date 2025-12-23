"""
CLI Integration Tests for IntentLog

These tests verify the CLI commands work correctly end-to-end.
"""

import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield Path(temp_dir)
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


def run_ilog(*args):
    """Helper to run ilog CLI commands"""
    result = subprocess.run(
        ["python", "-m", "intentlog.cli"] + list(args),
        capture_output=True,
        text=True,
    )
    return result


class TestInitCommand:
    """Tests for ilog init command"""

    def test_init_creates_project(self, temp_project_dir):
        """Test that init creates .intentlog directory"""
        result = run_ilog("init", "my-project")

        assert result.returncode == 0
        assert "Initialized IntentLog project: my-project" in result.stdout
        assert (temp_project_dir / ".intentlog").is_dir()
        assert (temp_project_dir / ".intentlog" / "config.json").is_file()

    def test_init_fails_if_exists(self, temp_project_dir):
        """Test that init fails if already initialized"""
        run_ilog("init", "project1")
        result = run_ilog("init", "project2")

        assert result.returncode == 1
        assert "Error" in result.stdout

    def test_init_force_reinitializes(self, temp_project_dir):
        """Test that init --force reinitializes"""
        run_ilog("init", "project1")
        result = run_ilog("init", "--force", "project2")

        assert result.returncode == 0
        assert "project2" in result.stdout


class TestCommitCommand:
    """Tests for ilog commit command"""

    def test_commit_adds_intent(self, temp_project_dir):
        """Test that commit adds an intent"""
        run_ilog("init", "test-project")
        result = run_ilog("commit", "We chose microservices for better scalability")

        assert result.returncode == 0
        assert "main:" in result.stdout

        # Verify intent was saved
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        with open(intents_file) as f:
            data = json.load(f)
        assert len(data["intents"]) == 1
        assert "microservices" in data["intents"][0]["intent_reasoning"]

    def test_commit_fails_without_init(self, temp_project_dir):
        """Test that commit fails if not initialized"""
        result = run_ilog("commit", "Some message")

        assert result.returncode == 1
        assert "Error" in result.stdout

    def test_commit_with_attach(self, temp_project_dir):
        """Test commit with --attach flag"""
        # Initialize git repo
        subprocess.run(["git", "init"], capture_output=True, cwd=temp_project_dir)

        # Create and stage a file
        test_file = temp_project_dir / "test.txt"
        test_file.write_text("test content")
        subprocess.run(["git", "add", "test.txt"], capture_output=True, cwd=temp_project_dir)

        run_ilog("init", "test-project")
        result = run_ilog("commit", "Adding test file", "--attach")

        assert result.returncode == 0
        # Check that files were attached in metadata
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        with open(intents_file) as f:
            data = json.load(f)
        assert "attached_files" in data["intents"][0]["metadata"]


class TestBranchCommand:
    """Tests for ilog branch command"""

    def test_branch_list_shows_main(self, temp_project_dir):
        """Test that branch list shows main"""
        run_ilog("init", "test-project")
        result = run_ilog("branch", "--list")

        assert result.returncode == 0
        assert "main" in result.stdout
        assert "*" in result.stdout  # Current branch marker

    def test_branch_creates_new_branch(self, temp_project_dir):
        """Test creating a new branch"""
        run_ilog("init", "test-project")
        result = run_ilog("branch", "feature-x")

        assert result.returncode == 0
        assert "Created and switched to branch 'feature-x'" in result.stdout

    def test_branch_switches_to_existing(self, temp_project_dir):
        """Test switching to existing branch"""
        run_ilog("init", "test-project")
        run_ilog("branch", "develop")
        run_ilog("branch", "main")  # Switch back to main
        result = run_ilog("branch", "develop")  # Switch to develop

        assert result.returncode == 0
        assert "Switched to branch 'develop'" in result.stdout


class TestLogCommand:
    """Tests for ilog log command"""

    def test_log_empty(self, temp_project_dir):
        """Test log with no intents"""
        run_ilog("init", "test-project")
        result = run_ilog("log")

        assert result.returncode == 0
        assert "No intents" in result.stdout

    def test_log_shows_intents(self, temp_project_dir):
        """Test log shows committed intents"""
        run_ilog("init", "test-project")
        run_ilog("commit", "First intent")
        run_ilog("commit", "Second intent")
        result = run_ilog("log")

        assert result.returncode == 0
        assert "First intent" in result.stdout
        assert "Second intent" in result.stdout

    def test_log_limit(self, temp_project_dir):
        """Test log with --limit"""
        run_ilog("init", "test-project")
        for i in range(5):
            run_ilog("commit", f"Intent {i}")

        result = run_ilog("log", "--limit", "2")

        assert result.returncode == 0
        assert "Intent 4" in result.stdout
        assert "Intent 3" in result.stdout
        assert "and 3 more" in result.stdout


class TestSearchCommand:
    """Tests for ilog search command"""

    def test_search_finds_match(self, temp_project_dir):
        """Test search finds matching intents"""
        run_ilog("init", "test-project")
        run_ilog("commit", "We chose microservices architecture")
        run_ilog("commit", "Fixed login bug")

        result = run_ilog("search", "microservices")

        assert result.returncode == 0
        assert "microservices" in result.stdout
        assert "login" not in result.stdout

    def test_search_no_match(self, temp_project_dir):
        """Test search with no matches"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Some intent")

        result = run_ilog("search", "nonexistent")

        assert result.returncode == 0
        assert "No intents matching" in result.stdout


class TestStatusCommand:
    """Tests for ilog status command"""

    def test_status_shows_info(self, temp_project_dir):
        """Test status shows project info"""
        run_ilog("init", "test-project")
        run_ilog("commit", "First intent")
        run_ilog("branch", "feature")

        result = run_ilog("status")

        assert result.returncode == 0
        assert "test-project" in result.stdout
        assert "feature" in result.stdout  # Current branch
        assert "Intents: 1" in result.stdout


class TestAuditCommand:
    """Tests for ilog audit command"""

    def test_audit_passes_valid(self, temp_project_dir):
        """Test audit passes with valid log"""
        log_file = temp_project_dir / "valid.log"
        log_file.write_text("intent_name: 'test'\nintent_reasoning: 'valid reasoning'")

        result = run_ilog("audit", str(log_file))

        assert result.returncode == 0
        assert "Passed" in result.stdout

    def test_audit_fails_empty_reasoning(self, temp_project_dir):
        """Test audit fails with empty reasoning"""
        log_file = temp_project_dir / "invalid.log"
        log_file.write_text("intent_name: 'test'\nintent_reasoning: ''")

        result = run_ilog("audit", str(log_file))

        assert result.returncode == 1
        assert "HALLUCINATION_RISK" in result.stdout

    def test_audit_file_not_found(self, temp_project_dir):
        """Test audit fails with missing file"""
        result = run_ilog("audit", "nonexistent.log")

        assert result.returncode == 1
        assert "not found" in result.stdout


class TestVersionFlag:
    """Tests for --version flag"""

    def test_version_shows_version(self, temp_project_dir):
        """Test --version shows version"""
        result = run_ilog("--version")

        assert result.returncode == 0
        assert "0.1.0" in result.stdout


class TestHelpFlag:
    """Tests for help"""

    def test_help_shows_commands(self, temp_project_dir):
        """Test help shows available commands"""
        result = run_ilog("--help")

        assert result.returncode == 0
        assert "init" in result.stdout
        assert "commit" in result.stdout
        assert "branch" in result.stdout
        assert "log" in result.stdout
        assert "search" in result.stdout
        assert "audit" in result.stdout

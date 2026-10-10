"""
CLI Integration Tests for IntentLog

These tests verify the CLI commands work correctly end-to-end.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from intentlog import CRYPTO_AVAILABLE


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
        assert "0.2.0" in result.stdout


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


def _latest_intent_id(branch=None):
    """Return the intent_id of the newest intent on a branch via 'ilog log --json'"""
    args = ["log", "--json"]
    if branch:
        args += ["--branch", branch]
    result = run_ilog(*args)
    assert result.returncode == 0, result.stdout
    return json.loads(result.stdout)[0]["intent_id"]


def _read_intents(intents_file):
    """Load a branch file and index its intents by intent_id"""
    with open(intents_file) as f:
        data = json.load(f)
    return {entry["intent_id"]: entry for entry in data["intents"]}


class TestShowCommand:
    """Tests for ilog show command"""

    def test_show_resolves_hash_printed_by_commit(self, temp_project_dir):
        """The hash printed by commit is the one log prints and show accepts"""
        run_ilog("init", "test-project")
        result = run_ilog("commit", "Chose PostgreSQL for concurrent writes")

        assert result.returncode == 0
        first_line = result.stdout.splitlines()[0]
        assert first_line.startswith("[")
        assert "] main:" in first_line
        printed_hash = first_line[1:first_line.index("]")]
        assert len(printed_hash) == 12

        log_result = run_ilog("log")
        assert f"[{printed_hash}]" in log_result.stdout

        show_result = run_ilog("show", printed_hash)
        assert show_result.returncode == 0
        assert f"Intent {printed_hash}" in show_result.stdout
        assert "PostgreSQL" in show_result.stdout

    def test_show_accepts_chain_hash_prefix(self, temp_project_dir):
        """A chain hash prefix (what older commit output printed) still resolves"""
        run_ilog("init", "test-project")
        run_ilog("commit", "First intent")

        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        with open(intents_file) as f:
            entry = json.load(f)["intents"][0]

        result = run_ilog("show", entry["chain_hash"][:12])

        assert result.returncode == 0
        assert f"ID:        {entry['intent_id']}" in result.stdout

    def test_show_unknown_id_fails(self, temp_project_dir):
        """Test show with an id that matches nothing"""
        run_ilog("init", "test-project")
        run_ilog("commit", "First intent")

        result = run_ilog("show", "ffffffffffff")

        assert result.returncode == 1
        assert "No intent matching" in result.stdout


class TestLinkCommand:
    """Tests for ilog link command"""

    def test_link_same_branch(self, temp_project_dir):
        """Linking two intents on the current branch persists the target id"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Target intent")
        target_id = _latest_intent_id()
        run_ilog("commit", "Source intent")
        source_id = _latest_intent_id()

        result = run_ilog("link", source_id[:8], target_id[:8])

        assert result.returncode == 0
        assert "Linked [" in result.stdout
        assert "Target intent" in result.stdout
        assert "(on branch" not in result.stdout

        by_id = _read_intents(temp_project_dir / ".intentlog" / "intents.json")
        assert by_id[source_id]["metadata"]["links"] == [target_id]

    def test_link_target_on_other_branch(self, temp_project_dir):
        """A link may point at an intent that only exists on another branch"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Initial decision")
        run_ilog("branch", "exploration")
        run_ilog("commit", "Exploration result")
        exploration_id = _latest_intent_id("exploration")
        run_ilog("branch", "main")
        run_ilog("commit", "Final decision")
        decision_id = _latest_intent_id("main")

        result = run_ilog("link", decision_id[:8], exploration_id[:8])

        assert result.returncode == 0, result.stdout
        assert "Error" not in result.stdout
        assert "Exploration result" in result.stdout
        assert "(on branch 'exploration')" in result.stdout

        # Persisted on the source (main) branch only
        main_by_id = _read_intents(temp_project_dir / ".intentlog" / "intents.json")
        assert main_by_id[decision_id]["metadata"]["links"] == [exploration_id]
        branch_by_id = _read_intents(
            temp_project_dir / ".intentlog" / "branches" / "exploration.json"
        )
        assert "links" not in branch_by_id[exploration_id]["metadata"]

        show_result = run_ilog("show", decision_id[:8])
        assert show_result.returncode == 0
        assert exploration_id[:12] in show_result.stdout

    def test_link_twice_reports_already_linked(self, temp_project_dir):
        """Linking the same pair twice does not duplicate the link"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Target intent")
        target_id = _latest_intent_id()
        run_ilog("commit", "Source intent")
        source_id = _latest_intent_id()

        run_ilog("link", source_id[:8], target_id[:8])
        result = run_ilog("link", source_id[:8], target_id[:8])

        assert result.returncode == 0
        assert "Already linked" in result.stdout
        by_id = _read_intents(temp_project_dir / ".intentlog" / "intents.json")
        assert by_id[source_id]["metadata"]["links"] == [target_id]

    def test_link_unknown_target_fails(self, temp_project_dir):
        """Test link fails when the target matches nothing on any branch"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Source intent")
        source_id = _latest_intent_id()

        result = run_ilog("link", source_id[:8], "ffffffffffff")

        assert result.returncode == 1
        assert "No intent matching 'ffffffffffff'" in result.stdout

    @pytest.mark.parametrize("bad_branch", ["../evil", "bad name"])
    def test_link_invalid_branch_name_fails_cleanly(self, temp_project_dir, bad_branch):
        """An invalid --branch name is reported as an error, not a traceback"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Source intent")
        source_id = _latest_intent_id()

        result = run_ilog("link", source_id[:8], source_id[:8], "--branch", bad_branch)

        assert result.returncode == 1
        assert "Error:" in result.stdout
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_link_unknown_branch_fails_cleanly(self, temp_project_dir):
        """A well-formed but missing --branch is reported as not found"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Source intent")
        source_id = _latest_intent_id()

        result = run_ilog("link", source_id[:8], source_id[:8], "--branch", "nope")

        assert result.returncode == 1
        assert "Error: Branch 'nope' not found" in result.stdout
        assert "Traceback" not in result.stderr


class TestTagCommand:
    """Tests for ilog tag command"""

    def test_tag_add_and_remove(self, temp_project_dir):
        """Tags added by tag persist and can be removed again"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Tagged intent")
        intent_id = _latest_intent_id()

        result = run_ilog("tag", intent_id[:8], "alpha", "beta")
        assert result.returncode == 0
        assert "tags: alpha, beta" in result.stdout

        result = run_ilog("tag", intent_id[:8], "alpha", "--remove")
        assert result.returncode == 0
        assert "Removed tag: alpha" in result.stdout

        by_id = _read_intents(temp_project_dir / ".intentlog" / "intents.json")
        assert by_id[intent_id]["metadata"]["tags"] == ["beta"]

    @pytest.mark.parametrize("bad_branch", ["../evil", "bad name"])
    def test_tag_invalid_branch_name_fails_cleanly(self, temp_project_dir, bad_branch):
        """An invalid --branch name is reported as an error, not a traceback"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Tagged intent")
        intent_id = _latest_intent_id()

        result = run_ilog("tag", intent_id[:8], "t", "--branch", bad_branch)

        assert result.returncode == 1
        assert "Error:" in result.stdout
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

        # Nothing was written
        by_id = _read_intents(temp_project_dir / ".intentlog" / "intents.json")
        assert "tags" not in by_id[intent_id]["metadata"]

    def test_tag_unknown_branch_fails_cleanly(self, temp_project_dir):
        """A well-formed but missing --branch is reported as not found"""
        run_ilog("init", "test-project")
        run_ilog("commit", "Tagged intent")
        intent_id = _latest_intent_id()

        result = run_ilog("tag", intent_id[:8], "t", "--branch", "nope")

        assert result.returncode == 1
        assert "Error: Branch 'nope' not found" in result.stdout
        assert "Traceback" not in result.stderr


# =========================================================================
# Chain data and signatures survive tag, link and merge
# =========================================================================


def _branch_entries(path):
    with open(path) as f:
        return json.load(f)["intents"]


class TestChainPreservedByCommands:
    """tag, link and merge keep the stored chain of untouched intents"""

    def test_tag_keeps_earlier_chain_entries(self, temp_project_dir):
        run_ilog("init", "test-project")
        for message in ("first", "second", "third"):
            run_ilog("commit", message)
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        before = _branch_entries(intents_file)

        result = run_ilog("tag", before[2]["intent_id"][:8], "x")
        assert result.returncode == 0, result.stdout

        after = _branch_entries(intents_file)
        assert after[:2] == before[:2]
        assert after[2]["prev_hash"] == before[1]["chain_hash"]
        verify = run_ilog("chain", "verify")
        assert verify.returncode == 0 and "PASSED" in verify.stdout

    def test_merge_keeps_target_chain(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("commit", "on main")
        run_ilog("branch", "feat")
        run_ilog("commit", "on feat")
        run_ilog("branch", "main")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        before = _branch_entries(intents_file)

        result = run_ilog("merge", "feat")
        assert result.returncode == 0, result.stdout

        after = _branch_entries(intents_file)
        assert after[0] == before[0]
        assert len(after) == 2 and all("chain_hash" in e for e in after)
        verify = run_ilog("chain", "verify")
        assert verify.returncode == 0 and "PASSED" in verify.stdout


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
class TestSignedAnnotations:
    """tag and link refuse to remove signatures unless --force is given"""

    def _signed_project(self, messages):
        run_ilog("init", "test-project")
        result = run_ilog("keys", "generate", "--name", "k")
        assert result.returncode == 0, result.stdout + result.stderr
        for message in messages:
            result = run_ilog("commit", "--sign", message)
            assert result.returncode == 0, result.stdout + result.stderr

    def test_tag_without_force_is_refused(self, temp_project_dir):
        self._signed_project(["signed"])
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        before = _branch_entries(intents_file)

        result = run_ilog("tag", before[0]["intent_id"][:8], "x")

        assert result.returncode == 1
        assert "--force" in result.stdout
        assert "Added tag" not in result.stdout
        assert "Traceback" not in result.stdout + result.stderr
        assert _branch_entries(intents_file) == before

    def test_tag_with_force_removes_signature(self, temp_project_dir):
        self._signed_project(["signed"])
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        intent_id = _branch_entries(intents_file)[0]["intent_id"]

        result = run_ilog("tag", intent_id[:8], "x", "--force")

        assert result.returncode == 0, result.stdout
        assert "removed 1 signature" in result.stdout
        entry = _branch_entries(intents_file)[0]
        assert entry["metadata"]["tags"] == ["x"]
        assert "signature" not in entry
        verify = run_ilog("chain", "verify")
        assert verify.returncode == 0 and "PASSED" in verify.stdout

    def test_link_without_force_is_refused(self, temp_project_dir):
        self._signed_project(["older", "newer"])
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        before = _branch_entries(intents_file)

        result = run_ilog("link", before[1]["intent_id"][:8], before[0]["intent_id"][:8])

        assert result.returncode == 1
        assert "--force" in result.stdout
        assert _branch_entries(intents_file) == before

    def test_branch_keeps_signatures(self, temp_project_dir):
        self._signed_project(["signed"])

        run_ilog("branch", "feat")

        feat = _branch_entries(temp_project_dir / ".intentlog" / "branches" / "feat.json")
        assert feat[0].get("signature")


def _damage_last_entry(intents_file):
    """Remove the chain data from the newest stored entry, as a hand edit would."""
    with open(intents_file) as f:
        data = json.load(f)
    for key in ("intent_hash", "prev_hash", "chain_hash", "sequence"):
        data["intents"][-1].pop(key, None)
    with open(intents_file, "w") as f:
        json.dump(data, f, indent=2)


class TestDamagedChainCommands:
    """Commands that would rewrite a damaged chain refuse unless --force"""

    def test_merge_into_damaged_branch_needs_force(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("commit", "first")
        run_ilog("commit", "second")
        run_ilog("branch", "feat")
        run_ilog("commit", "on feat")
        run_ilog("branch", "main")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        _damage_last_entry(intents_file)
        before = _branch_entries(intents_file)

        result = run_ilog("merge", "feat")
        assert result.returncode == 1
        assert "--force" in result.stdout
        assert "Traceback" not in result.stdout + result.stderr
        assert _branch_entries(intents_file) == before

        result = run_ilog("merge", "feat", "--force")
        assert result.returncode == 0, result.stdout

    def test_chain_migrate_on_damaged_branch_needs_force(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("commit", "first")
        run_ilog("commit", "second")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        _damage_last_entry(intents_file)

        result = run_ilog("chain", "migrate")
        assert result.returncode == 1
        assert "--force" in result.stdout
        assert run_ilog("chain", "verify").returncode == 1

        result = run_ilog("chain", "migrate", "--force")
        assert result.returncode == 0, result.stdout
        assert run_ilog("chain", "verify").returncode == 0


class TestRepairAllUnreadable:
    def test_migrate_force_repairs_branch_with_only_unreadable_entries(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("commit", "only entry")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        with open(intents_file) as f:
            data = json.load(f)
        data["intents"][0] = None
        with open(intents_file, "w") as f:
            json.dump(data, f)

        assert run_ilog("commit", "refused").returncode == 1
        assert run_ilog("chain", "migrate").returncode == 1

        result = run_ilog("chain", "migrate", "--force")
        assert result.returncode == 0, result.stdout
        assert run_ilog("chain", "verify").returncode == 0
        assert run_ilog("commit", "works again").returncode == 0


class TestTagNoOp:
    def test_removing_absent_tag_rewrites_nothing(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("commit", "first")
        run_ilog("commit", "second")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        before = _branch_entries(intents_file)

        result = run_ilog("tag", before[0]["intent_id"][:8], "nothere", "--remove")

        assert result.returncode == 0, result.stdout
        assert "Tag not found: nothere" in result.stdout
        assert _branch_entries(intents_file) == before

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_adding_existing_tag_on_signed_chain_needs_no_force(self, temp_project_dir):
        run_ilog("init", "test-project")
        run_ilog("keys", "generate", "--name", "k")
        intents_file = temp_project_dir / ".intentlog" / "intents.json"
        run_ilog("commit", "--sign", "signed")
        intent_id = _branch_entries(intents_file)[0]["intent_id"]
        assert run_ilog("tag", intent_id[:8], "x", "--force").returncode == 0
        before = _branch_entries(intents_file)

        result = run_ilog("tag", intent_id[:8], "x")

        assert result.returncode == 0, result.stdout
        assert "Already tagged: x" in result.stdout
        assert _branch_entries(intents_file) == before


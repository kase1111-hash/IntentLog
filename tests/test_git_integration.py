"""
Tests for IntentLog git integration.

Tests the git helper module (intentlog.git) and the git-aware
CLI commands (init, commit, log, status, show, hooks, blame).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from intentlog.git import (
    detect_git_repo,
    get_git_context,
    get_git_branch,
    get_git_head,
    get_git_log_for_file,
    install_hooks,
    uninstall_hooks,
    hooks_status,
    HOOK_MARKER,
)
from intentlog.storage import IntentLogStorage, ProjectConfig


def _git(cwd, *args):
    """Run a git command with signing disabled."""
    env = os.environ.copy()
    env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_COMMITTER_EMAIL"] = "test@test.com"
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with an initial commit."""
    _git(tmp_path, "init", str(tmp_path))
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    # Create initial commit
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\n")
    _git(tmp_path, "add", ".")
    result = _git(tmp_path, "commit", "-m", "Initial commit")
    assert result.returncode == 0, f"git commit failed: {result.stderr}"
    return tmp_path


@pytest.fixture
def ilog_in_git(git_repo):
    """Create an IntentLog project inside a git repo."""
    storage = IntentLogStorage(project_path=git_repo)
    config = storage.init_project("test-project", force=True)
    config.git_root = str(git_repo)
    storage.save_config(config)
    return storage, git_repo


# =========================================================================
# git.py unit tests
# =========================================================================


class TestDetectGitRepo:
    def test_detects_git_repo(self, git_repo):
        result = detect_git_repo(git_repo)
        assert result is not None
        assert result == git_repo

    def test_detects_from_subdirectory(self, git_repo):
        subdir = git_repo / "src"
        subdir.mkdir()
        result = detect_git_repo(subdir)
        assert result == git_repo

    def test_returns_none_for_non_git(self, tmp_path):
        result = detect_git_repo(tmp_path)
        assert result is None


class TestGetGitContext:
    def test_captures_branch(self, git_repo):
        ctx = get_git_context(git_repo)
        # Default branch might be main or master depending on git config
        assert "branch" in ctx
        assert ctx["branch"] in ("main", "master")

    def test_captures_head_commit(self, git_repo):
        ctx = get_git_context(git_repo)
        assert "commit" in ctx
        assert len(ctx["commit"]) == 40  # Full SHA

    def test_captures_staged_files(self, git_repo):
        # Stage a new file
        new_file = git_repo / "test.txt"
        new_file.write_text("hello")
        _git(git_repo, "add", "test.txt")

        ctx = get_git_context(git_repo)
        assert "staged_files" in ctx
        assert "test.txt" in ctx["staged_files"]

    def test_empty_staged_when_nothing_staged(self, git_repo):
        ctx = get_git_context(git_repo)
        assert ctx["staged_files"] == []


class TestGetGitBranch:
    def test_returns_branch_name(self, git_repo):
        branch = get_git_branch(git_repo)
        assert branch in ("main", "master")

    def test_returns_none_for_non_git(self, tmp_path):
        branch = get_git_branch(tmp_path)
        assert branch is None


class TestGetGitHead:
    def test_returns_commit_hash(self, git_repo):
        head = get_git_head(git_repo)
        assert head is not None
        assert len(head) == 40

    def test_returns_none_for_non_git(self, tmp_path):
        head = get_git_head(tmp_path)
        assert head is None


class TestGetGitLogForFile:
    def test_returns_log_entries(self, git_repo):
        entries = get_git_log_for_file(git_repo, "README.md")
        assert len(entries) >= 1
        assert entries[0]["message"] == "Initial commit"
        assert len(entries[0]["commit"]) == 40

    def test_returns_empty_for_nonexistent_file(self, git_repo):
        entries = get_git_log_for_file(git_repo, "nonexistent.txt")
        assert entries == []

    def test_tracks_multiple_commits(self, git_repo):
        # Make a second commit
        readme = git_repo / "README.md"
        readme.write_text("# Updated\n")
        _git(git_repo, "add", ".")
        result = _git(git_repo, "commit", "-m", "Update README")
        assert result.returncode == 0

        entries = get_git_log_for_file(git_repo, "README.md")
        assert len(entries) == 2
        assert entries[0]["message"] == "Update README"
        assert entries[1]["message"] == "Initial commit"


# =========================================================================
# Hook management tests
# =========================================================================


class TestHookManagement:
    def test_install_hooks(self, git_repo):
        installed = install_hooks(git_repo)
        assert len(installed) >= 1
        assert any("prepare-commit-msg" in h for h in installed)

        hook_path = git_repo / ".git" / "hooks" / "prepare-commit-msg"
        assert hook_path.exists()
        content = hook_path.read_text()
        assert HOOK_MARKER in content

    def test_install_hooks_idempotent(self, git_repo):
        install_hooks(git_repo)
        installed2 = install_hooks(git_repo)
        assert len(installed2) >= 1  # Should update, not fail

    def test_install_refuses_foreign_hook(self, git_repo):
        # Write a foreign hook
        hook_path = git_repo / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# My custom hook\necho hello\n")
        hook_path.chmod(0o755)

        installed = install_hooks(git_repo)
        assert installed == []  # Should refuse to overwrite

    def test_uninstall_hooks(self, git_repo):
        install_hooks(git_repo)
        removed = uninstall_hooks(git_repo)
        assert "prepare-commit-msg" in removed

        hook_path = git_repo / ".git" / "hooks" / "prepare-commit-msg"
        assert not hook_path.exists()

    def test_uninstall_leaves_foreign_hooks(self, git_repo):
        # Write a foreign hook
        hook_path = git_repo / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\n# My custom hook\necho hello\n")
        hook_path.chmod(0o755)

        removed = uninstall_hooks(git_repo)
        assert removed == []
        assert hook_path.exists()  # Should not be removed

    def test_hooks_status(self, git_repo):
        status = hooks_status(git_repo)
        assert status["prepare-commit-msg"] == "not installed"

        install_hooks(git_repo)
        status = hooks_status(git_repo)
        assert status["prepare-commit-msg"] == "installed"

    def test_hooks_status_detects_foreign(self, git_repo):
        hook_path = git_repo / ".git" / "hooks" / "prepare-commit-msg"
        hook_path.write_text("#!/bin/sh\necho foreign\n")
        hook_path.chmod(0o755)

        status = hooks_status(git_repo)
        assert status["prepare-commit-msg"] == "foreign"

    def test_install_hooks_no_git(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            install_hooks(tmp_path)


# =========================================================================
# Storage integration tests
# =========================================================================


class TestProjectConfigGitRoot:
    def test_git_root_in_config(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        config = storage.load_config()
        assert config.git_root == str(git_repo)

    def test_git_root_roundtrips(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        config = storage.load_config()

        # Save and reload
        storage.save_config(config)
        config2 = storage.load_config()
        assert config2.git_root == str(git_repo)

    def test_git_root_in_to_dict(self, git_repo):
        config = ProjectConfig(project_name="test", git_root=str(git_repo))
        d = config.to_dict()
        assert d["git_root"] == str(git_repo)

    def test_git_root_omitted_when_empty(self):
        config = ProjectConfig(project_name="test")
        d = config.to_dict()
        assert "git_root" not in d

    def test_git_root_from_dict_missing(self):
        config = ProjectConfig.from_dict({"project_name": "test"})
        assert config.git_root == ""


# =========================================================================
# Intent with git context tests
# =========================================================================


class TestIntentGitContext:
    def test_commit_captures_git_context(self, ilog_in_git):
        storage, git_repo = ilog_in_git

        chained = storage.add_chained_intent(
            name="Test intent",
            reasoning="Testing git context capture",
            metadata={"git_context": get_git_context(git_repo)},
        )

        # Reload and verify
        intents = storage.load_intents()
        assert len(intents) == 1
        git_ctx = intents[0].metadata.get("git_context", {})
        assert "branch" in git_ctx
        assert "commit" in git_ctx
        assert len(git_ctx["commit"]) == 40

    def test_search_by_git_commit(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        git_ctx = get_git_context(git_repo)

        storage.add_chained_intent(
            name="First intent",
            reasoning="First reasoning",
            metadata={"git_context": git_ctx},
        )

        # Search by commit hash prefix
        intents = storage.load_intents()
        commit_hash = git_ctx["commit"]
        matched = [
            i for i in intents
            if i.metadata.get("git_context", {}).get("commit", "").startswith(commit_hash[:8])
        ]
        assert len(matched) == 1
        assert matched[0].intent_name == "First intent"

    def test_multiple_intents_different_commits(self, ilog_in_git):
        storage, git_repo = ilog_in_git

        # First intent with current HEAD
        git_ctx1 = get_git_context(git_repo)
        storage.add_chained_intent(
            name="Before change",
            reasoning="Before modifying README",
            metadata={"git_context": git_ctx1},
        )

        # Make a new git commit
        readme = git_repo / "README.md"
        readme.write_text("# Updated\n")
        _git(git_repo, "add", ".")
        result = _git(git_repo, "commit", "-m", "Update README")
        assert result.returncode == 0

        # Second intent with new HEAD
        git_ctx2 = get_git_context(git_repo)
        storage.add_chained_intent(
            name="After change",
            reasoning="After modifying README",
            metadata={"git_context": git_ctx2},
        )

        intents = storage.load_intents()
        assert len(intents) == 2
        assert intents[0].metadata["git_context"]["commit"] != intents[1].metadata["git_context"]["commit"]


# =========================================================================
# CLI command integration tests
# =========================================================================


class TestCLIGitIntegration:
    """Test the CLI commands with git integration."""

    def _run_ilog(self, *args, cwd=None):
        """Run ilog CLI command and return (returncode, stdout, stderr)."""
        result = subprocess.run(
            ["python", "-m", "intentlog.cli"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def test_init_detects_git(self, git_repo):
        rc, out, _ = self._run_ilog("init", "myproject", cwd=git_repo)
        assert rc == 0
        assert "Git repo:" in out

        # Verify config has git_root
        storage = IntentLogStorage(project_path=git_repo)
        config = storage.load_config()
        assert config.git_root == str(git_repo)

    def test_init_without_git(self, tmp_path):
        rc, out, _ = self._run_ilog("init", "myproject", cwd=tmp_path)
        assert rc == 0
        assert "Git repo:" not in out

    def test_commit_shows_git_context(self, ilog_in_git):
        storage, git_repo = ilog_in_git

        # Stage a file so git context has staged files
        test_file = git_repo / "test.py"
        test_file.write_text("print('hello')\n")
        _git(git_repo, "add", "test.py")

        rc, out, _ = self._run_ilog("commit", "Adding test file", cwd=git_repo)
        assert rc == 0
        assert "Git:" in out
        assert "Staged:" in out

    def test_log_shows_git_context(self, ilog_in_git):
        storage, git_repo = ilog_in_git

        # Create an intent with git context
        git_ctx = get_git_context(git_repo)
        storage.add_chained_intent(
            name="Test log display",
            reasoning="Testing that log shows git info",
            metadata={"git_context": git_ctx},
        )

        rc, out, _ = self._run_ilog("log", cwd=git_repo)
        assert rc == 0
        assert "git:" in out

    def test_log_json_output(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        storage.add_chained_intent(
            name="JSON test",
            reasoning="Testing JSON output",
        )

        rc, out, _ = self._run_ilog("log", "--json", cwd=git_repo)
        assert rc == 0
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["intent_name"] == "JSON test"

    def test_log_git_commit_filter(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        git_ctx = get_git_context(git_repo)

        storage.add_chained_intent(
            name="Linked intent",
            reasoning="Has git context",
            metadata={"git_context": git_ctx},
        )
        storage.add_chained_intent(
            name="Unlinked intent",
            reasoning="No git context",
        )

        commit_prefix = git_ctx["commit"][:8]
        rc, out, _ = self._run_ilog("log", "--git-commit", commit_prefix, cwd=git_repo)
        assert rc == 0
        assert "Linked intent" in out
        assert "Unlinked intent" not in out

    def test_status_shows_git(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("status", cwd=git_repo)
        assert rc == 0
        assert "Git:" in out

    def test_status_json(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("status", "--json", cwd=git_repo)
        assert rc == 0
        data = json.loads(out)
        assert "git" in data
        assert "branch" in data["git"]

    def test_show_intent(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        git_ctx = get_git_context(git_repo)
        chained = storage.add_chained_intent(
            name="Detailed intent",
            reasoning="This has a lot of detail to show",
            metadata={"git_context": git_ctx},
        )

        # Show by intent ID prefix
        intent_id_prefix = chained.intent.intent_id[:8]
        rc, out, _ = self._run_ilog("show", intent_id_prefix, cwd=git_repo)
        assert rc == 0
        assert "Detailed intent" in out
        assert "Git Context:" in out

    def test_show_intent_json(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="JSON show test",
            reasoning="Testing JSON show output",
        )

        intent_id_prefix = chained.intent.intent_id[:8]
        rc, out, _ = self._run_ilog("show", intent_id_prefix, "--json", cwd=git_repo)
        assert rc == 0
        data = json.loads(out)
        assert data["intent_name"] == "JSON show test"

    def test_show_nonexistent(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("show", "nonexistent123", cwd=git_repo)
        assert rc != 0

    def test_hooks_install_and_status(self, ilog_in_git):
        _, git_repo = ilog_in_git

        # Check status before install
        rc, out, _ = self._run_ilog("hooks", "status", cwd=git_repo)
        assert rc == 0
        assert "not installed" in out

        # Install
        rc, out, _ = self._run_ilog("hooks", "install", cwd=git_repo)
        assert rc == 0
        assert "installed" in out.lower()

        # Check status after install
        rc, out, _ = self._run_ilog("hooks", "status", cwd=git_repo)
        assert rc == 0
        assert "installed" in out

    def test_hooks_uninstall(self, ilog_in_git):
        _, git_repo = ilog_in_git

        self._run_ilog("hooks", "install", cwd=git_repo)
        rc, out, _ = self._run_ilog("hooks", "uninstall", cwd=git_repo)
        assert rc == 0
        assert "Removed" in out or "removed" in out.lower()

    def test_blame_command(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        git_ctx = get_git_context(git_repo)

        # Record an intent linked to the commit that created README.md
        storage.add_chained_intent(
            name="Document project setup",
            reasoning="Creating initial project documentation",
            metadata={"git_context": git_ctx},
        )

        rc, out, _ = self._run_ilog("blame", "README.md", cwd=git_repo)
        assert rc == 0
        assert "Initial commit" in out

    def test_blame_no_intents(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("blame", "README.md", cwd=git_repo)
        assert rc == 0
        assert "no intent recorded" in out

    def test_blame_nonexistent_file(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("blame", "nonexistent.txt", cwd=git_repo)
        assert rc == 0
        assert "No git history" in out

    # =====================================================================
    # Tag command tests
    # =====================================================================

    def test_tag_add_single(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Taggable intent", reasoning="Testing tag add",
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("tag", prefix, "architecture", cwd=git_repo)
        assert rc == 0
        assert "Added tag: architecture" in out

        # Verify persisted
        intents = storage.load_intents()
        assert "architecture" in intents[0].metadata.get("tags", [])

    def test_tag_add_multiple(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Multi-tag intent", reasoning="Testing multiple tags",
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("tag", prefix, "design", "database", "v2", cwd=git_repo)
        assert rc == 0
        assert "Added tag: design" in out
        assert "Added tag: database" in out
        assert "Added tag: v2" in out

        intents = storage.load_intents()
        tags = intents[0].metadata.get("tags", [])
        assert set(tags) == {"design", "database", "v2"}

    def test_tag_add_duplicate_skipped(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Dup tag intent", reasoning="Testing duplicate tags",
            metadata={"tags": ["existing"]},
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("tag", prefix, "existing", cwd=git_repo)
        assert rc == 0
        assert "Already tagged: existing" in out

    def test_tag_remove(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Remove tag intent", reasoning="Testing tag removal",
            metadata={"tags": ["keep", "remove-me"]},
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("tag", prefix, "remove-me", "--remove", cwd=git_repo)
        assert rc == 0
        assert "Removed tag: remove-me" in out

        intents = storage.load_intents()
        tags = intents[0].metadata.get("tags", [])
        assert "keep" in tags
        assert "remove-me" not in tags

    def test_tag_remove_nonexistent(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="No such tag", reasoning="Testing removal of missing tag",
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("tag", prefix, "nope", "--remove", cwd=git_repo)
        assert rc == 0
        assert "Tag not found: nope" in out

    def test_tag_shows_in_log(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Tagged in log", reasoning="Testing tag display in log",
            metadata={"tags": ["visible-tag"]},
        )

        rc, out, _ = self._run_ilog("log", cwd=git_repo)
        assert rc == 0
        assert "visible-tag" in out

    def test_tag_shows_in_show(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        chained = storage.add_chained_intent(
            name="Tagged in show", reasoning="Testing tag display in show",
            metadata={"tags": ["show-tag", "another"]},
        )
        prefix = chained.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("show", prefix, cwd=git_repo)
        assert rc == 0
        assert "Tags:" in out
        assert "show-tag" in out
        assert "another" in out

    def test_tag_nonexistent_intent(self, ilog_in_git):
        _, git_repo = ilog_in_git
        rc, out, _ = self._run_ilog("tag", "nonexistent99", "foo", cwd=git_repo)
        assert rc != 0

    # =====================================================================
    # Link command tests
    # =====================================================================

    def test_link_two_intents(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        c1 = storage.add_chained_intent(
            name="Decision", reasoning="Chose PostgreSQL over MongoDB",
        )
        c2 = storage.add_chained_intent(
            name="Exploration", reasoning="Tried MongoDB but it was too slow",
        )
        src = c1.intent.intent_id[:8]
        tgt = c2.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("link", src, tgt, cwd=git_repo)
        assert rc == 0
        assert "Linked" in out
        assert "Decision" in out
        assert "Exploration" in out

        # Verify persisted
        intents = storage.load_intents()
        source_intent = [i for i in intents if i.intent_id.startswith(src)][0]
        links = source_intent.metadata.get("links", [])
        assert c2.intent.intent_id in links

    def test_link_duplicate_skipped(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        c1 = storage.add_chained_intent(name="A", reasoning="First")
        c2 = storage.add_chained_intent(name="B", reasoning="Second")
        src = c1.intent.intent_id[:8]
        tgt = c2.intent.intent_id[:8]

        # Link once
        self._run_ilog("link", src, tgt, cwd=git_repo)
        # Link again
        rc, out, _ = self._run_ilog("link", src, tgt, cwd=git_repo)
        assert rc == 0
        assert "Already linked" in out

    def test_link_shows_in_show(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        c1 = storage.add_chained_intent(name="Source", reasoning="Source intent")
        c2 = storage.add_chained_intent(name="Target", reasoning="Target intent")

        src = c1.intent.intent_id[:8]
        tgt = c2.intent.intent_id[:8]
        self._run_ilog("link", src, tgt, cwd=git_repo)

        rc, out, _ = self._run_ilog("show", src, cwd=git_repo)
        assert rc == 0
        assert "Linked intents:" in out

    def test_link_nonexistent_source(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        c1 = storage.add_chained_intent(name="Exists", reasoning="Real intent")
        tgt = c1.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("link", "nonexistent99", tgt, cwd=git_repo)
        assert rc != 0

    def test_link_nonexistent_target(self, ilog_in_git):
        storage, git_repo = ilog_in_git
        c1 = storage.add_chained_intent(name="Exists", reasoning="Real intent")
        src = c1.intent.intent_id[:8]

        rc, out, _ = self._run_ilog("link", src, "nonexistent99", cwd=git_repo)
        assert rc != 0

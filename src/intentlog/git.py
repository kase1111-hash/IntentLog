"""
Git integration for IntentLog.

Provides helper functions for detecting git repositories, capturing
git context (branch, HEAD, staged files), and managing git hooks.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List


def _run_git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    """
    Run a git command and return stdout, or None on failure.

    Args:
        *args: git subcommand and arguments (e.g., "rev-parse", "--show-toplevel")
        cwd: Working directory for the command

    Returns:
        Stripped stdout string, or None if git is not available or command fails
    """
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_git_repo(path: Optional[Path] = None) -> Optional[Path]:
    """
    Detect if the given path is inside a git repository.

    Args:
        path: Directory to check. Defaults to cwd.

    Returns:
        Path to the git repo root, or None if not in a git repo.
    """
    cwd = path or Path.cwd()
    toplevel = _run_git("rev-parse", "--show-toplevel", cwd=cwd)
    if toplevel:
        return Path(toplevel)
    return None


def get_git_context(git_root: Path) -> Dict[str, Any]:
    """
    Capture the current git context for an intent commit.

    Collects:
    - Current branch name
    - HEAD commit hash
    - List of staged files (from git diff --cached)

    Args:
        git_root: Path to the git repository root

    Returns:
        Dict with git context data. Empty dict if git info unavailable.
    """
    context: Dict[str, Any] = {}

    # Current branch
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=git_root)
    if branch:
        context["branch"] = branch

    # HEAD commit hash
    head = _run_git("rev-parse", "HEAD", cwd=git_root)
    if head:
        context["commit"] = head

    # Staged files
    staged = _run_git("diff", "--cached", "--name-only", cwd=git_root)
    if staged:
        context["staged_files"] = [f for f in staged.split("\n") if f]
    else:
        context["staged_files"] = []

    return context


def get_git_branch(git_root: Path) -> Optional[str]:
    """Get the current git branch name."""
    return _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=git_root)


def get_git_head(git_root: Path) -> Optional[str]:
    """Get the current git HEAD commit hash."""
    return _run_git("rev-parse", "HEAD", cwd=git_root)


def get_git_log_for_file(
    git_root: Path,
    file_path: str,
    limit: int = 50,
) -> List[Dict[str, str]]:
    """
    Get git log entries for a specific file.

    Args:
        git_root: Path to the git repository root
        file_path: File path relative to git root
        limit: Maximum number of log entries

    Returns:
        List of dicts with 'commit', 'author', 'date', 'message' keys
    """
    output = _run_git(
        "log",
        f"-{limit}",
        "--format=%H%n%an%n%ai%n%s%n---",
        "--follow",
        "--",
        file_path,
        cwd=git_root,
    )
    if not output:
        return []

    entries = []
    for block in output.split("\n---"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 3)
        if len(lines) >= 4:
            entries.append({
                "commit": lines[0],
                "author": lines[1],
                "date": lines[2],
                "message": lines[3].strip(),
            })
    return entries


# =========================================================================
# Git Hook Management
# =========================================================================

PREPARE_COMMIT_MSG_HOOK = """\
#!/bin/sh
# IntentLog prepare-commit-msg hook
# Prepends the latest intent summary to the git commit message.
#
# Installed by: ilog hooks install
# Remove by deleting this file or running: ilog hooks uninstall

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"

# Only modify if this is a normal commit (not merge, squash, etc.)
if [ -z "$COMMIT_SOURCE" ] || [ "$COMMIT_SOURCE" = "message" ]; then
    # Get the latest intent from IntentLog
    INTENT=$(intentlog log --limit 1 --json 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$INTENT" ]; then
        # Prepend intent info to commit message
        TEMP_FILE=$(mktemp)
        echo "" >> "$TEMP_FILE"
        echo "# IntentLog: Latest recorded intent" >> "$TEMP_FILE"
        echo "$INTENT" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
        cat "$COMMIT_MSG_FILE" >> "$TEMP_FILE"
        mv "$TEMP_FILE" "$COMMIT_MSG_FILE"
    fi
fi
"""

POST_COMMIT_HOOK = """\
#!/bin/sh
# IntentLog post-commit hook
# Records the git commit hash against the most recent intent.
#
# Installed by: ilog hooks install
# Remove by deleting this file or running: ilog hooks uninstall

# This is informational only — the intent was already recorded.
# Future versions may use this to link intents to commits retroactively.
"""

HOOK_MARKER = "# Installed by: ilog hooks install"


def get_hooks_dir(git_root: Path) -> Optional[Path]:
    """Get the git hooks directory."""
    hooks_dir = git_root / ".git" / "hooks"
    if hooks_dir.is_dir():
        return hooks_dir
    return None


def install_hooks(git_root: Path) -> List[str]:
    """
    Install IntentLog git hooks.

    Args:
        git_root: Path to the git repository root

    Returns:
        List of installed hook names

    Raises:
        FileNotFoundError: If .git/hooks directory doesn't exist
    """
    hooks_dir = get_hooks_dir(git_root)
    if not hooks_dir:
        raise FileNotFoundError(f"Git hooks directory not found at {git_root / '.git' / 'hooks'}")

    installed = []

    # Install prepare-commit-msg hook
    hook_path = hooks_dir / "prepare-commit-msg"
    if hook_path.exists():
        content = hook_path.read_text()
        if HOOK_MARKER in content:
            # Already installed by us — overwrite
            hook_path.write_text(PREPARE_COMMIT_MSG_HOOK)
            installed.append("prepare-commit-msg (updated)")
        else:
            # User has their own hook — don't overwrite
            return []
    else:
        hook_path.write_text(PREPARE_COMMIT_MSG_HOOK)
        hook_path.chmod(0o755)
        installed.append("prepare-commit-msg")

    return installed


def uninstall_hooks(git_root: Path) -> List[str]:
    """
    Uninstall IntentLog git hooks.

    Only removes hooks that were installed by IntentLog (identified by marker).

    Args:
        git_root: Path to the git repository root

    Returns:
        List of removed hook names
    """
    hooks_dir = get_hooks_dir(git_root)
    if not hooks_dir:
        return []

    removed = []
    for hook_name in ["prepare-commit-msg", "post-commit"]:
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            content = hook_path.read_text()
            if HOOK_MARKER in content:
                hook_path.unlink()
                removed.append(hook_name)

    return removed


def hooks_status(git_root: Path) -> Dict[str, str]:
    """
    Check the status of IntentLog git hooks.

    Returns:
        Dict mapping hook name to status ("installed", "foreign", "not installed")
    """
    hooks_dir = get_hooks_dir(git_root)
    if not hooks_dir:
        return {}

    status = {}
    for hook_name in ["prepare-commit-msg", "post-commit"]:
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            content = hook_path.read_text()
            if HOOK_MARKER in content:
                status[hook_name] = "installed"
            else:
                status[hook_name] = "foreign"
        else:
            status[hook_name] = "not installed"

    return status

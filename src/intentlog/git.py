"""
Git integration for IntentLog.

Provides helper functions for detecting git repositories, capturing
git context (branch, HEAD, staged files), and managing git hooks.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


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

HOOK_BLOCK_START = "# >>> IntentLog: latest recorded intent (removed on commit)"
HOOK_BLOCK_END = "# <<< IntentLog"

PREPARE_COMMIT_MSG_HOOK = """\
#!/bin/sh
# IntentLog prepare-commit-msg hook
# Shows the latest recorded intent in the commit message editor, as a marked
# block of comment lines. The companion IntentLog commit-msg hook deletes the
# block before git stores the message, so the stored message never changes.
#
# Installed by: ilog hooks install
# Remove by deleting this file or running: ilog hooks uninstall

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"

# Only plain commits. Leave -m/-F ("message"), templates, merge, squash and
# amend ("commit") messages untouched.
[ -z "$COMMIT_SOURCE" ] || exit 0

# Git writes its own '#' help lines only when it is about to open the editor,
# so their absence means --no-edit or commit.status=false: stay out.
grep -q '^#' "$COMMIT_MSG_FILE" 2>/dev/null || exit 0

# Only add the block when the companion commit-msg hook will remove it again.
grep -q 'IntentLog commit-msg hook' "$(dirname "$0")/commit-msg" 2>/dev/null || exit 0

# Stay out when git keeps comment lines, in case commit-msg is skipped
# (git commit --no-verify).
case "$(git config --get commit.cleanup 2>/dev/null)" in
    whitespace|verbatim|scissors) exit 0 ;;
esac
for key in core.commentChar core.commentString; do
    case "$(git config --get "$key" 2>/dev/null)" in
        ""|"#") ;;
        *) exit 0 ;;
    esac
done

# Bail out quietly when IntentLog is not installed.
command -v intentlog >/dev/null 2>&1 || exit 0

# Get the latest intent as plain text. Fails silently when there is no
# IntentLog project here.
INTENT=$(intentlog log --limit 1 2>/dev/null) || exit 0
[ -n "$INTENT" ] || exit 0

{
    printf '%s\\n' '__START__'
    printf '%s\\n' "$INTENT" | sed 's/^/# /'
    printf '%s\\n' '__END__'
} >> "$COMMIT_MSG_FILE"
""".replace("__START__", HOOK_BLOCK_START).replace("__END__", HOOK_BLOCK_END)

COMMIT_MSG_HOOK = """\
#!/bin/sh
# IntentLog commit-msg hook
# Deletes the block the IntentLog prepare-commit-msg hook added, so it never
# ends up in the stored commit message, whatever git's cleanup mode. A block
# whose end marker is missing, or that contains a line not starting with '#',
# is left alone so that nothing the user typed is ever removed.
#
# Installed by: ilog hooks install
# Remove by deleting this file or running: ilog hooks uninstall

MSG_FILE="$1"
grep -qxF '__START__' "$MSG_FILE" 2>/dev/null || exit 0

TMP_FILE="$MSG_FILE.intentlog.$$"
if awk -v start='__START__' -v end='__END__' '
    !inblk && $0 == start { inblk = 1; ok = 1; n = 0; buf[++n] = $0; next }
    inblk {
        buf[++n] = $0
        if ($0 == end) {
            if (!ok) for (i = 1; i <= n; i++) print buf[i]
            inblk = 0
            next
        }
        if (substr($0, 1, 1) != "#") ok = 0
        next
    }
    { print }
    END { if (inblk) for (i = 1; i <= n; i++) print buf[i] }
' "$MSG_FILE" > "$TMP_FILE"; then
    mv "$TMP_FILE" "$MSG_FILE"
else
    rm -f "$TMP_FILE"
fi
exit 0
""".replace("__START__", HOOK_BLOCK_START).replace("__END__", HOOK_BLOCK_END)

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


# Hooks managed by `ilog hooks`, with the body each one should have.
MANAGED_HOOKS = {
    "prepare-commit-msg": PREPARE_COMMIT_MSG_HOOK,
    "commit-msg": COMMIT_MSG_HOOK,
}


def install_hooks(git_root: Path) -> List[str]:
    """
    Install IntentLog git hooks.

    Installs a ``prepare-commit-msg`` hook that shows the latest recorded
    intent in the commit message editor, and a ``commit-msg`` hook that
    removes that block again before git stores the message, so the stored
    commit message is never changed. Nothing is installed if either hook
    already exists and was not installed by IntentLog.

    Args:
        git_root: Path to the git repository root

    Returns:
        List of installed hook names (empty if a foreign hook is in the way)

    Raises:
        FileNotFoundError: If .git/hooks directory doesn't exist
    """
    hooks_dir = get_hooks_dir(git_root)
    if not hooks_dir:
        raise FileNotFoundError(f"Git hooks directory not found at {git_root / '.git' / 'hooks'}")

    # Never overwrite a hook the user wrote themselves
    for hook_name in MANAGED_HOOKS:
        hook_path = hooks_dir / hook_name
        if hook_path.exists() and HOOK_MARKER not in hook_path.read_text():
            return []

    installed = []
    # commit-msg first: prepare-commit-msg only acts when it is present
    for hook_name in ("commit-msg", "prepare-commit-msg"):
        hook_path = hooks_dir / hook_name
        existed = hook_path.exists()
        hook_path.write_text(MANAGED_HOOKS[hook_name])
        hook_path.chmod(0o755)
        installed.append(f"{hook_name} (updated)" if existed else hook_name)

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
    for hook_name in ["prepare-commit-msg", "commit-msg", "post-commit"]:
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
        Dict mapping hook name to status: "installed", "outdated" (installed
        by an older IntentLog; re-run 'ilog hooks install'), "foreign" or
        "not installed"
    """
    hooks_dir = get_hooks_dir(git_root)
    if not hooks_dir:
        return {}

    expected = dict(MANAGED_HOOKS)
    expected["post-commit"] = POST_COMMIT_HOOK

    status = {}
    for hook_name in ["prepare-commit-msg", "commit-msg", "post-commit"]:
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            content = hook_path.read_text()
            if HOOK_MARKER not in content:
                status[hook_name] = "foreign"
            elif content == expected[hook_name]:
                status[hook_name] = "installed"
            else:
                status[hook_name] = "outdated"
        else:
            status[hook_name] = "not installed"

    return status

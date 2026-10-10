"""
IntentLog Storage Module

This module provides persistent storage for IntentLog projects.
Manages the .intentlog/ directory structure, JSON serialization,
and file locking for concurrent access.

Supports:
- Merkle tree hash chains for tamper-evident history
- Ed25519 cryptographic signatures
- Branch management with chain verification
"""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from .core import Intent
from .logging import get_logger, log_context
from .validation import ValidationError, validate_branch_name, validate_project_name

logger = get_logger(__name__)


def _atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Atomically write JSON data to a file.

    Writes to a temporary file in the same directory, then renames it
    into place. os.replace() is atomic on POSIX systems, so a crash
    mid-write will leave either the old file or the new file, never a
    partial write.
    """
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".intentlog_")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _parse_datetime(value: str, field_name: str = "timestamp") -> datetime:
    """
    Safely parse an ISO format datetime string.

    Args:
        value: The datetime string to parse
        field_name: Name of the field being parsed (for error messages)

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the datetime string is invalid
    """
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {field_name} format '{value}': {e}") from e

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .crypto import KeyManager
    from .merkle import ChainedIntent, ChainVerificationResult


INTENTLOG_DIR = ".intentlog"
CONFIG_FILE = "config.json"
INTENTS_FILE = "intents.json"
BRANCHES_DIR = "branches"


@dataclass
class LLMSettings:
    """LLM configuration for semantic features"""
    provider: str = ""  # "openai", "anthropic", "ollama"
    model: str = ""  # Model name
    api_key_env: str = ""  # Environment variable for API key
    embedding_provider: str = ""  # Provider for embeddings (if different)
    embedding_model: str = ""  # Embedding model name
    base_url: str = ""  # Custom API endpoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "base_url": self.base_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMSettings":
        return cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            api_key_env=data.get("api_key_env", ""),
            embedding_provider=data.get("embedding_provider", ""),
            embedding_model=data.get("embedding_model", ""),
            base_url=data.get("base_url", ""),
        )

    def is_configured(self) -> bool:
        """Check if LLM is configured"""
        return bool(self.provider)


@dataclass
class ProjectConfig:
    """Configuration for an IntentLog project"""
    project_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_branch: str = "main"
    version: str = "0.2.0"
    llm: LLMSettings = field(default_factory=LLMSettings)
    git_root: str = ""  # Path to git repo root, if detected

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "project_name": self.project_name,
            "created_at": self.created_at,
            "current_branch": self.current_branch,
            "version": self.version,
            "llm": self.llm.to_dict(),
        }
        if self.git_root:
            d["git_root"] = self.git_root
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        llm_data = data.get("llm", {})
        return cls(
            project_name=data["project_name"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            current_branch=data.get("current_branch", "main"),
            version=data.get("version", "0.1.0"),
            llm=LLMSettings.from_dict(llm_data) if llm_data else LLMSettings(),
            git_root=data.get("git_root", ""),
        )


class StorageError(Exception):
    """Base exception for storage errors"""
    pass


class ProjectNotFoundError(StorageError):
    """Raised when no .intentlog directory is found"""
    pass


class ProjectExistsError(StorageError):
    """Raised when trying to init in existing project"""
    pass


class BranchNotFoundError(StorageError):
    """Raised when specified branch doesn't exist"""
    pass


class BranchExistsError(StorageError):
    """Raised when trying to create existing branch"""
    pass


class ChainIntegrityError(StorageError):
    """Raised when a write would rewrite stored chain entries that fail verification"""
    pass


class SignatureLossError(StorageError):
    """Raised when a write would remove intent signatures and that was not allowed"""

    def __init__(self, branch: str, count: int):
        self.branch = branch
        self.count = count
        super().__init__(
            f"Saving branch '{branch}' would remove {count} signature(s). Signed "
            "intents are hashed together with their metadata, so changing an "
            "intent re-chains it and every intent after it"
        )


def compute_intent_hash(intent: Intent) -> str:
    """
    Compute SHA-256 hash of an intent for integrity verification.

    Uses canonical JSON (sorted keys, no extra whitespace) to ensure
    consistent hashes regardless of serialization order.
    """
    # Create canonical representation
    canonical = {
        "intent_id": intent.intent_id,
        "intent_name": intent.intent_name,
        "intent_reasoning": intent.intent_reasoning,
        "timestamp": intent.timestamp.isoformat() if isinstance(intent.timestamp, datetime) else intent.timestamp,
        "parent_intent_id": intent.parent_intent_id,
        "metadata": intent.metadata,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:12]


def find_project_root(start_path: Optional[Path] = None, max_depth: int = 20) -> Optional[Path]:
    """
    Find the .intentlog directory by searching up from start_path.

    Args:
        start_path: Starting directory (default: cwd)
        max_depth: Maximum number of parent directories to traverse

    Returns the path containing .intentlog, or None if not found.
    """
    current = Path(start_path or os.getcwd()).resolve()

    depth = 0
    while current != current.parent and depth < max_depth:
        if (current / INTENTLOG_DIR).is_dir():
            return current
        current = current.parent
        depth += 1

    # Check final level (filesystem root)
    if depth <= max_depth and (current / INTENTLOG_DIR).is_dir():
        return current

    return None


def _placeholder_intent(entry: Dict[str, Any]) -> Intent:
    """Best-effort Intent for a stored entry that cannot be parsed."""
    try:
        timestamp = _parse_datetime(entry.get("timestamp"))
    except ValueError:
        timestamp = datetime(1970, 1, 1)
    metadata = entry.get("metadata")
    parent = entry.get("parent_intent_id")
    return Intent(
        intent_id=str(entry.get("intent_id", "")),
        intent_name=str(entry.get("intent_name", "")),
        intent_reasoning=str(entry.get("intent_reasoning", "")),
        timestamp=timestamp,
        metadata=metadata if isinstance(metadata, dict) else {},
        parent_intent_id=parent if isinstance(parent, str) else None,
    )


def _is_chained_data(data: Dict[str, Any]) -> bool:
    """True if raw branch data carries hash-chain information."""
    if "chain_version" in data:
        return True
    return any(
        isinstance(entry, dict) and "chain_hash" in entry
        for entry in data.get("intents", [])
    )


class IntentLogStorage:
    """
    Manages persistent storage for IntentLog projects.

    Directory structure:
        .intentlog/
        ├── config.json          # Project configuration
        ├── intents.json         # Main branch intents
        └── branches/
            ├── feature-x.json   # Branch-specific intents
            └── experiment.json
    """

    #: Seconds to wait for a branch lock before raising FileLockTimeout.
    lock_timeout: float = 10.0

    def __init__(self, project_path: Optional[Path] = None):
        """
        Initialize storage for a project.

        Args:
            project_path: Path to project root. If None, searches up from cwd.
        """
        if project_path:
            self.project_root = Path(project_path).resolve()
        else:
            found = find_project_root()
            if found:
                self.project_root = found
            else:
                self.project_root = Path.cwd().resolve()

        self.intentlog_dir = self.project_root / INTENTLOG_DIR
        self.config_path = self.intentlog_dir / CONFIG_FILE
        self.branches_dir = self.intentlog_dir / BRANCHES_DIR

    def _get_branch_file(self, branch_name: str) -> Path:
        """
        Get the path to a branch's intent file.

        Raises:
            BranchNotFoundError: If the name is not a valid branch name, so a
                traversal-style name can never address a file outside
                .intentlog/
        """
        try:
            validate_branch_name(branch_name)
        except ValidationError as e:
            raise BranchNotFoundError(f"Branch '{branch_name}' not found") from e
        if branch_name == "main":
            return self.intentlog_dir / INTENTS_FILE
        return self.branches_dir / f"{branch_name}.json"

    def _get_lock_file(self, file_path: Path) -> Path:
        """
        Get the lock file path for a data file under .intentlog/.

        Locks live in .intentlog/temp/locks/, mirroring the data file's path
        (e.g. temp/locks/branches/feature.json.lock). They are taken on a
        separate, stable file rather than on the data file itself, because
        _atomic_write_json() replaces data files with os.replace(), which
        would leave lock holders attached to an unlinked inode. temp/ is in
        the .gitignore that 'ilog init' writes, so locks never show up as
        untracked files.
        """
        relative = Path(file_path).relative_to(self.intentlog_dir)
        if ".." in relative.parts:
            raise StorageError(f"Refusing to lock a path outside {INTENTLOG_DIR}/: {file_path}")
        return self.intentlog_dir / "temp" / "locks" / relative.parent / (relative.name + ".lock")

    def _lock_file(self, file_path: Path, exclusive: bool = True):
        """
        Context manager for file locking.

        Uses cross-platform file locking to prevent concurrent writes.
        Works on Unix (fcntl), Windows (msvcrt), and falls back to lock files.
        Locks are re-entrant within a thread (see filelock.FileLock).
        """
        from .filelock import FileLock
        return FileLock(
            self._get_lock_file(file_path),
            exclusive=exclusive,
            timeout=self.lock_timeout,
        )

    @contextmanager
    def lock_branch(self, branch: Optional[str] = None) -> Iterator[str]:
        """
        Hold an exclusive lock on a branch for a read-modify-write sequence.

        Use this around any load -> mutate -> save sequence so that no other
        thread or process can interleave its own write and cause a lost
        update. The lock is re-entrant within the current thread, so
        load_intents(), save_intents(), add_intent() and the chained
        variants can be called freely inside the block without deadlocking.

        Args:
            branch: Branch to lock. If None, uses current branch.

        Yields:
            The name of the locked branch (resolved from config when None).

        Raises:
            ProjectNotFoundError: If project not initialized
            ValidationError: If the branch name is invalid
            FileLockTimeout: If the lock cannot be acquired within lock_timeout

        Usage:
            with storage.lock_branch() as branch:
                intents = storage.load_intents(branch)
                intents.append(new_intent)
                storage.save_intents(intents, branch)
        """
        # Resolve and validate BEFORE locking: acquiring the lock creates the
        # lock file (and its parent directories), so an uninitialized project
        # must raise first, or a stray .intentlog/ would shadow the real
        # project root; likewise a traversal-style name, whether passed in or
        # read from config.json, must never reach _get_branch_file().
        config = self.load_config()
        branch = validate_branch_name(branch or config.current_branch)
        with self._lock_file(self._get_branch_file(branch), exclusive=True):
            yield branch

    def is_initialized(self) -> bool:
        """Check if project is initialized"""
        return self.intentlog_dir.is_dir() and self.config_path.is_file()

    def init_project(self, project_name: str, force: bool = False) -> ProjectConfig:
        """
        Initialize a new IntentLog project.

        Creates:
        - .intentlog/ directory
        - config.json with project settings
        - Empty intents.json for main branch
        - branches/ directory
        - .gitignore for sensitive data

        Args:
            project_name: Name for the project
            force: If True, reinitialize existing project

        Returns:
            ProjectConfig for the new project

        Raises:
            ProjectExistsError: If project exists and force=False
            ValidationError: If project name is invalid
        """
        logger = get_logger()

        # Validate project name to prevent path traversal
        project_name = validate_project_name(project_name)

        if self.is_initialized() and not force:
            logger.debug("Project already initialized", path=str(self.project_root))
            raise ProjectExistsError(
                f"IntentLog already initialized in {self.project_root}"
            )

        logger.info("Initializing IntentLog project", project=project_name, path=str(self.project_root))

        # Create directory structure
        self.intentlog_dir.mkdir(exist_ok=True)
        self.branches_dir.mkdir(exist_ok=True)
        logger.debug("Created directory structure")

        # Create config
        config = ProjectConfig(project_name=project_name)
        _atomic_write_json(self.config_path, config.to_dict())

        # Create empty main branch intents
        main_intents_path = self._get_branch_file("main")
        _atomic_write_json(main_intents_path, {"intents": [], "branch": "main"})

        # Create .gitignore for sensitive data
        gitignore_path = self.intentlog_dir / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write("# IntentLog sensitive data — never commit!\n")
            f.write("*.key\n")
            f.write("*.pem\n")
            f.write("*.secret\n")
            f.write("keys.json\n")
            f.write("violations.jsonl\n")
            f.write("network_audit.jsonl\n")
            f.write("temp/\n")

        logger.info("Project initialized successfully", project=project_name)
        return config

    def load_config(self) -> ProjectConfig:
        """
        Load project configuration.

        Returns:
            ProjectConfig for the project

        Raises:
            ProjectNotFoundError: If project not initialized
        """
        if not self.is_initialized():
            raise ProjectNotFoundError(
                "No IntentLog project found. Run 'ilog init' first."
            )

        with open(self.config_path) as f:
            data = json.load(f)

        return ProjectConfig.from_dict(data)

    def save_config(self, config: ProjectConfig) -> None:
        """Save project configuration"""
        _atomic_write_json(self.config_path, config.to_dict())

    def load_intents(self, branch: Optional[str] = None) -> List[Intent]:
        """
        Load intents from a branch.

        Args:
            branch: Branch name. If None, uses current branch.

        Returns:
            List of Intent objects

        Raises:
            ProjectNotFoundError: If project not initialized
            BranchNotFoundError: If branch doesn't exist
        """
        config = self.load_config()
        branch = branch or config.current_branch

        branch_file = self._get_branch_file(branch)
        if not branch_file.is_file():
            if branch == "main":
                return []
            raise BranchNotFoundError(f"Branch '{branch}' not found")

        with self._lock_file(branch_file, exclusive=False):
            with open(branch_file) as f:
                data = json.load(f)

        intents = []
        for intent_data in data.get("intents", []):
            try:
                intent = Intent(
                    intent_id=intent_data["intent_id"],
                    intent_name=intent_data["intent_name"],
                    intent_reasoning=intent_data["intent_reasoning"],
                    timestamp=_parse_datetime(intent_data["timestamp"]),
                    metadata=intent_data.get("metadata", {}),
                    parent_intent_id=intent_data.get("parent_intent_id"),
                )
                intents.append(intent)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid intent data: {e}")
                continue

        return intents

    def save_intents(
        self,
        intents: List[Intent],
        branch: Optional[str] = None,
        *,
        strict: bool = False,
    ) -> int:
        """
        Save intents to a branch.

        On a hash-chained branch the stored chain is preserved: every stored
        entry up to the first intent that was changed, inserted or removed is
        kept verbatim, hashes and signature included, and only the intents
        from that point on are re-chained. Appending intents therefore never
        rewrites history. Changing an earlier intent (for example tagging it,
        since tags live in the hashed metadata) re-chains that intent and every
        later one, which invalidates their signatures, so those signatures are
        removed. Branches without chain data keep the plain format.

        Args:
            intents: List of Intent objects to save
            branch: Branch name. If None, uses current branch.
            strict: If True, refuse a rewrite that would remove signatures or
                overwrite stored entries that fail chain verification. If
                False, proceed and log a warning.

        Returns:
            Number of signatures removed because their intents were re-chained.

        Raises:
            SignatureLossError: If strict and signatures would be removed
            ChainIntegrityError: If strict and the part of the stored chain
                being rewritten fails verification
        """
        config = self.load_config()
        branch = branch or config.current_branch

        branch_file = self._get_branch_file(branch)

        with self._lock_file(branch_file, exclusive=True):
            existing = self._read_branch_data(branch_file)

            if _is_chained_data(existing):
                chained, removed = self._rechain_preserving(
                    branch, existing, intents, strict=strict
                )
                self.save_chained_intents(chained, branch)
                return removed

            data = {
                "branch": branch,
                "intents": [
                    {
                        "intent_id": i.intent_id,
                        "intent_name": i.intent_name,
                        "intent_reasoning": i.intent_reasoning,
                        "timestamp": i.timestamp.isoformat() if isinstance(i.timestamp, datetime) else i.timestamp,
                        "metadata": i.metadata,
                        "parent_intent_id": i.parent_intent_id,
                        "hash": compute_intent_hash(i),
                    }
                    for i in intents
                ],
            }
            _atomic_write_json(branch_file, data)
            return 0

    def _read_branch_data(self, branch_file: Path) -> Dict[str, Any]:
        """Read a branch file's raw JSON under a shared lock ({} if it does not exist)."""
        if not branch_file.is_file():
            return {}
        with self._lock_file(branch_file, exclusive=False):
            with open(branch_file) as f:
                data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _rechain_preserving(
        self,
        branch: str,
        existing: Dict[str, Any],
        intents: List[Intent],
        strict: bool = False,
        repair: bool = False,
    ) -> Tuple[List["ChainedIntent"], int]:
        """
        Build the chain for ``intents`` while keeping the stored chain intact.

        Stored entries are kept verbatim for as long as the intents match them
        (same position, same intent_id, same content). From the first mismatch
        on, intents are re-chained. Stored entries are kept even if they fail
        verification, so a write never hides tampering; a rewrite that would
        replace such entries is reported instead.

        With ``repair``, stored entries are only kept up to the first one that
        fails verification; the chain is rebuilt from there.

        Returns:
            (chain, number of signatures removed by the rewrite)
        """
        from .merkle import (
            GENESIS_HASH,
            ChainedIntent,
            append_to_chain,
            compute_chain_hash,
            compute_intent_content_hash,
        )

        logger = get_logger()
        raw_entries = list(existing.get("intents", []))

        # Parse the stored chain up to the first entry that is not a valid
        # chained entry; positions then line up with raw_entries.
        stored: List[ChainedIntent] = []
        for entry in raw_entries:
            if not isinstance(entry, dict) or "chain_hash" not in entry:
                break
            try:
                stored.append(ChainedIntent.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                break

        def entry_verifies(i: int) -> bool:
            cur = stored[i]
            expected_prev = stored[i - 1].chain_hash if i > 0 else GENESIS_HASH
            return (
                bool(cur.chain_hash)
                and compute_intent_content_hash(cur.intent) == cur.intent_hash
                and cur.prev_hash == expected_prev
                and compute_chain_hash(cur.intent_hash, cur.prev_hash) == cur.chain_hash
            )

        keep = 0
        while (
            keep < len(intents)
            and keep < len(stored)
            and stored[keep].intent.intent_id == intents[keep].intent_id
            and compute_intent_content_hash(stored[keep].intent)
            == compute_intent_content_hash(intents[keep])
            and (not repair or entry_verifies(keep))
        ):
            keep += 1

        # Stored entries from `keep` on are about to be replaced.
        branch_is_chained = _is_chained_data(existing)
        broken_at = None
        for i in range(keep, len(raw_entries)):
            if not isinstance(raw_entries[i], dict) or "chain_hash" not in raw_entries[i]:
                if branch_is_chained:
                    # On a chained branch every entry must carry chain data;
                    # a missing one means it was stripped by hand.
                    broken_at = i
                    break
                continue  # legacy plain branch: nothing to verify
            if i >= len(stored) or not entry_verifies(i):
                broken_at = i
                break

        if broken_at is not None:
            message = (
                f"Branch '{branch}' fails chain verification at position {broken_at}; "
                "rewriting it would hide the break. Run 'ilog chain verify' for details"
            )
            if strict:
                raise ChainIntegrityError(message)
            logger.warning(message)

        removed = sum(1 for e in raw_entries[keep:] if isinstance(e, dict) and e.get("signature"))
        if removed:
            if strict:
                raise SignatureLossError(branch, removed)
            logger.warning(
                "Removed signatures invalidated by re-chaining", branch=branch, count=removed
            )

        chained = list(stored[:keep])
        for intent in intents[keep:]:
            chained.append(append_to_chain(chained, intent))
        return chained, removed

    def add_intent(
        self,
        name: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Intent:
        """
        Add a new intent to the log.

        Args:
            name: Intent name/title
            reasoning: The reasoning/explanation
            metadata: Optional metadata dict
            parent_id: Optional parent intent ID
            branch: Branch to add to. If None, uses current branch.

        Returns:
            The created Intent with computed hash
        """
        logger = get_logger()

        intent = Intent(
            intent_name=name,
            intent_reasoning=reasoning,
            metadata=metadata or {},
            parent_intent_id=parent_id,
        )

        if not intent.validate():
            logger.warning("Invalid intent: missing name or reasoning")
            raise ValueError("Intent must have name and non-empty reasoning")

        with log_context(intent_id=intent.intent_id, branch=branch, operation="add_intent"):
            # Hold one exclusive lock across load + append + save so that
            # concurrent writers cannot both read N intents and write N+1.
            with self.lock_branch(branch) as locked_branch:
                intents = self.load_intents(locked_branch)
                intents.append(intent)
                self.save_intents(intents, locked_branch)
            logger.info("Added intent", name=name[:50], intent_id=intent.intent_id[:12])

        return intent

    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> None:
        """
        Create a new branch.

        Args:
            branch_name: Name for the new branch
            from_branch: Branch to copy from. If None, uses current branch.

        Raises:
            BranchExistsError: If branch already exists
            ValidationError: If branch name is invalid
        """
        logger = get_logger()

        # Validate branch name to prevent path traversal
        branch_name = validate_branch_name(branch_name)

        config = self.load_config()
        from_branch = from_branch or config.current_branch

        new_branch_file = self._get_branch_file(branch_name)

        # Lock the new branch so the exists-check and the create are atomic
        # with respect to concurrent create_branch/add_intent on that name.
        with self.lock_branch(branch_name):
            if new_branch_file.is_file():
                logger.debug("Branch already exists", branch=branch_name)
                raise BranchExistsError(f"Branch '{branch_name}' already exists")

            logger.info("Creating branch", branch=branch_name, from_branch=from_branch)

            # Copy intents from source branch (raises if it does not exist)
            intents = self.load_intents(from_branch)

            data: Dict[str, Any] = {
                "branch": branch_name,
                "created_from": from_branch,
                "created_at": datetime.now().isoformat(),
            }

            source = self._read_branch_data(self._get_branch_file(from_branch))
            if _is_chained_data(source):
                # Keep chain links and signatures: the new branch starts as an
                # exact copy of the stored source chain, damage included.
                for key in ("chain_version", "root_hash"):
                    if key in source:
                        data[key] = source[key]
                data["intents"] = list(source.get("intents", []))
            else:
                data["intents"] = [
                    {
                        "intent_id": i.intent_id,
                        "intent_name": i.intent_name,
                        "intent_reasoning": i.intent_reasoning,
                        "timestamp": i.timestamp.isoformat() if isinstance(i.timestamp, datetime) else i.timestamp,
                        "metadata": i.metadata,
                        "parent_intent_id": i.parent_intent_id,
                        "hash": compute_intent_hash(i),
                    }
                    for i in intents
                ]

            _atomic_write_json(new_branch_file, data)

        logger.info("Branch created", branch=branch_name, intents_copied=len(intents))

    def switch_branch(self, branch_name: str) -> None:
        """
        Switch to a different branch.

        Args:
            branch_name: Branch to switch to

        Raises:
            BranchNotFoundError: If branch doesn't exist
            ValidationError: If branch name is invalid
        """
        # Validate branch name to prevent path traversal
        branch_name = validate_branch_name(branch_name)

        branch_file = self._get_branch_file(branch_name)
        if not branch_file.is_file() and branch_name != "main":
            raise BranchNotFoundError(f"Branch '{branch_name}' not found")

        config = self.load_config()
        config.current_branch = branch_name
        self.save_config(config)

    def list_branches(self) -> List[str]:
        """List all branches"""
        branches = ["main"]

        if self.branches_dir.is_dir():
            for f in self.branches_dir.iterdir():
                if f.suffix == ".json":
                    branches.append(f.stem)

        return sorted(branches)

    def search_intents(
        self,
        query: str,
        branch: Optional[str] = None,
    ) -> List[Intent]:
        """
        Search intents by name or reasoning content.

        Args:
            query: Search query (case-insensitive)
            branch: Branch to search. If None, uses current branch.

        Returns:
            List of matching Intent objects
        """
        intents = self.load_intents(branch)
        query_lower = query.lower()

        results = []
        for intent in intents:
            if (query_lower in intent.intent_name.lower() or
                query_lower in intent.intent_reasoning.lower()):
                results.append(intent)

        return results

    def get_attached_files(self) -> List[str]:
        """
        Get list of files tracked by git (for --attach flag).

        Returns:
            List of file paths relative to project root
        """
        import subprocess

        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            files = result.stdout.strip().split('\n')
            return [f for f in files if f]  # Filter empty strings
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def get_file_hashes(self, files: List[str]) -> Dict[str, str]:
        """
        Compute SHA-256 hashes for a list of files.

        Args:
            files: List of file paths relative to project root

        Returns:
            Dict mapping file paths to their hashes
        """
        hashes = {}
        for file_path in files:
            full_path = self.project_root / file_path
            if full_path.is_file():
                with open(full_path, 'rb') as f:
                    content = f.read()
                hashes[file_path] = hashlib.sha256(content).hexdigest()[:12]
        return hashes

    # =========================================================================
    # Merkle Chain Methods
    # =========================================================================

    def load_chained_intents(
        self,
        branch: Optional[str] = None,
    ) -> List["ChainedIntent"]:
        """
        Load intents with full chain data.

        Args:
            branch: Branch name. If None, uses current branch.

        Returns:
            List of ChainedIntent objects with chain linking
        """
        from .merkle import ChainedIntent, chain_intents

        config = self.load_config()
        branch = branch or config.current_branch

        branch_file = self._get_branch_file(branch)
        if not branch_file.is_file():
            if branch == "main":
                return []
            raise BranchNotFoundError(f"Branch '{branch}' not found")

        with self._lock_file(branch_file, exclusive=False):
            with open(branch_file) as f:
                data = json.load(f)

        branch_is_chained = _is_chained_data(data)
        chained = []
        for intent_data in data.get("intents", []):
            try:
                # Check if chain data exists
                if isinstance(intent_data, dict) and "chain_hash" in intent_data:
                    # Load with existing chain data
                    chained.append(ChainedIntent.from_dict(intent_data))
                else:
                    # Legacy data without chain - need to rebuild
                    intent = Intent(
                        intent_id=intent_data["intent_id"],
                        intent_name=intent_data["intent_name"],
                        intent_reasoning=intent_data["intent_reasoning"],
                        timestamp=_parse_datetime(intent_data["timestamp"]),
                        metadata=intent_data.get("metadata", {}),
                        parent_intent_id=intent_data.get("parent_intent_id"),
                    )
                    chained.append(intent)
            except (KeyError, ValueError, TypeError) as e:
                if branch_is_chained:
                    # On a chained branch an entry that cannot be read is
                    # damage, not noise: keep it as a broken entry so
                    # verification reports it and no later write drops it
                    logger.warning(f"Unreadable chained intent data: {e}")
                    chained.append(ChainedIntent(
                        intent=_placeholder_intent(
                            intent_data if isinstance(intent_data, dict) else {}
                        ),
                        intent_hash="",
                        prev_hash="",
                        chain_hash="",
                        sequence=len(chained),
                    ))
                    continue
                logger.warning(f"Skipping invalid chained intent data: {e}")
                continue

        if any(isinstance(c, Intent) for c in chained):
            if not _is_chained_data(data):
                # Legacy branch recorded without chain data: build the chain
                chained = chain_intents(chained)
            else:
                # A chained branch with entries whose chain data was removed.
                # Keep them in place with empty hashes so verification reports
                # the break, wherever it is, instead of re-hashing it away.
                chained = [
                    c if not isinstance(c, Intent)
                    else ChainedIntent(intent=c, intent_hash="", prev_hash="", chain_hash="", sequence=i)
                    for i, c in enumerate(chained)
                ]

        return chained

    def save_chained_intents(
        self,
        chained_intents: List["ChainedIntent"],
        branch: Optional[str] = None,
    ) -> None:
        """
        Save chained intents with full chain data.

        Args:
            chained_intents: List of ChainedIntent objects
            branch: Branch name. If None, uses current branch.
        """
        from .merkle import compute_root_hash

        config = self.load_config()
        branch = branch or config.current_branch

        branch_file = self._get_branch_file(branch)

        data = {
            "branch": branch,
            "chain_version": "1.0",
            "root_hash": compute_root_hash(chained_intents),
            "intents": [c.to_dict() for c in chained_intents],
        }

        with self._lock_file(branch_file, exclusive=True):
            _atomic_write_json(branch_file, data)

    def add_chained_intent(
        self,
        name: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        branch: Optional[str] = None,
        sign: bool = False,
        key_password: Optional[str] = None,
    ) -> "ChainedIntent":
        """
        Add a new intent with chain linking.

        Args:
            name: Intent name/title
            reasoning: The reasoning/explanation
            metadata: Optional metadata dict
            parent_id: Optional parent intent ID
            branch: Branch to add to. If None, uses current branch.
            sign: If True, sign the intent with default key
            key_password: Password for signing key if encrypted

        Returns:
            The created ChainedIntent with chain links
        """
        from .merkle import append_to_chain

        intent = Intent(
            intent_name=name,
            intent_reasoning=reasoning,
            metadata=metadata or {},
            parent_intent_id=parent_id,
        )

        if not intent.validate():
            raise ValueError("Intent must have name and non-empty reasoning")

        # Hold one exclusive lock across load + append + save so that
        # concurrent writers cannot build two chains from the same prev_hash.
        with self.lock_branch(branch) as locked_branch:
            # Load existing chain
            chained_intents = self.load_chained_intents(locked_branch)
            damaged = [c.sequence for c in chained_intents if not c.chain_hash]
            if damaged:
                raise ChainIntegrityError(
                    f"Branch '{locked_branch}' has entries without chain data "
                    f"(position {damaged[0]}). Run 'ilog chain verify' for details; "
                    "'ilog chain migrate --force' rebuilds the chain from that point"
                )

            # Append to chain
            new_chained = append_to_chain(chained_intents, intent)

            # Sign if requested
            if sign:
                signature = self._sign_chained_intent(new_chained, key_password)
                new_chained.signature = signature

            chained_intents.append(new_chained)
            self.save_chained_intents(chained_intents, locked_branch)

        return new_chained

    def _sign_chained_intent(
        self,
        chained: "ChainedIntent",
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sign a chained intent with the project's default key."""
        from .crypto import KeyManager, sign_intent

        key_manager = KeyManager(self.intentlog_dir)
        if not key_manager.has_keys():
            raise StorageError("No signing keys available. Run 'ilog keys generate' first.")

        # Create signable data (content + chain links)
        signable = {
            "intent_id": chained.intent.intent_id,
            "intent_hash": chained.intent_hash,
            "prev_hash": chained.prev_hash,
            "chain_hash": chained.chain_hash,
            "sequence": chained.sequence,
        }

        signature = sign_intent(signable, key_manager, password=password)
        return signature.to_dict()

    def verify_chain(
        self,
        branch: Optional[str] = None,
    ) -> "ChainVerificationResult":
        """
        Verify the integrity of a branch's intent chain.

        Args:
            branch: Branch to verify. If None, uses current branch.

        Returns:
            ChainVerificationResult with detailed status
        """
        from .merkle import verify_chain

        chained = self.load_chained_intents(branch)
        return verify_chain(chained)

    def get_root_hash(self, branch: Optional[str] = None) -> str:
        """
        Get the root hash of a branch's chain.

        Args:
            branch: Branch name. If None, uses current branch.

        Returns:
            Root hash string
        """
        from .merkle import GENESIS_HASH, compute_root_hash

        chained = self.load_chained_intents(branch)
        if not chained:
            return GENESIS_HASH
        return compute_root_hash(chained)

    def get_inclusion_proof(
        self,
        sequence: int,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an inclusion proof for an intent.

        Args:
            sequence: Sequence number of intent
            branch: Branch name. If None, uses current branch.

        Returns:
            Proof dictionary
        """
        from .merkle import generate_inclusion_proof

        chained = self.load_chained_intents(branch)
        return generate_inclusion_proof(chained, sequence)

    def migrate_to_chain(self, branch: Optional[str] = None, force: bool = False) -> int:
        """
        Migrate legacy intents to chain format.

        Chain data already stored is kept as it is. If part of the stored
        chain would have to be rewritten because it is unreadable or was
        stripped, the migration is refused unless forced, because the rewrite
        would hide that damage from 'ilog chain verify'.

        Args:
            branch: Branch to migrate. If None, uses current branch.
            force: Rewrite damaged chain data anyway (removing any signatures
                on the rewritten entries).

        Returns:
            Number of intents migrated

        Raises:
            ChainIntegrityError: If not forced and damaged chain data would be
                rewritten
            SignatureLossError: If not forced and signatures would be removed
        """
        with self.lock_branch(branch) as locked_branch:
            intents = self.load_intents(locked_branch)
            if not intents:
                return 0

            # Chain them, keeping any chain data (and signatures) already stored
            existing = self._read_branch_data(self._get_branch_file(locked_branch))
            chained, _ = self._rechain_preserving(
                locked_branch, existing, intents, strict=not force, repair=force
            )

            # Save with chain data
            self.save_chained_intents(chained, locked_branch)

        return len(chained)

    # =========================================================================
    # Key Management Methods
    # =========================================================================

    def get_key_manager(self) -> "KeyManager":
        """Get the KeyManager for this project."""
        from .crypto import KeyManager
        return KeyManager(self.intentlog_dir)

    def has_signing_keys(self) -> bool:
        """Check if signing keys are configured."""
        try:
            km = self.get_key_manager()
            return km.has_keys()
        except Exception:
            return False

    def verify_intent_signature(
        self,
        chained: "ChainedIntent",
    ) -> bool:
        """
        Verify a chained intent's signature.

        Args:
            chained: ChainedIntent with signature

        Returns:
            True if signature is valid

        Raises:
            SignatureError: If signature is invalid
            KeyNotFoundError: If signing key not found
        """
        from .crypto import KeyManager, Signature, verify_intent_signature

        if not chained.signature:
            raise StorageError("Intent has no signature")

        key_manager = KeyManager(self.intentlog_dir)
        signature = Signature.from_dict(chained.signature)

        signable = {
            "intent_id": chained.intent.intent_id,
            "intent_hash": chained.intent_hash,
            "prev_hash": chained.prev_hash,
            "chain_hash": chained.chain_hash,
            "sequence": chained.sequence,
        }

        return verify_intent_signature(signable, signature, key_manager)

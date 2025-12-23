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

import json
import os
import hashlib
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from .core import Intent, IntentLog

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .merkle import ChainedIntent, ChainVerificationResult
    from .crypto import KeyManager, Signature


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
    version: str = "0.1.0"
    llm: LLMSettings = field(default_factory=LLMSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "created_at": self.created_at,
            "current_branch": self.current_branch,
            "version": self.version,
            "llm": self.llm.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        llm_data = data.get("llm", {})
        return cls(
            project_name=data["project_name"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            current_branch=data.get("current_branch", "main"),
            version=data.get("version", "0.1.0"),
            llm=LLMSettings.from_dict(llm_data) if llm_data else LLMSettings(),
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


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the .intentlog directory by searching up from start_path.

    Returns the path containing .intentlog, or None if not found.
    """
    current = Path(start_path or os.getcwd()).resolve()

    while current != current.parent:
        if (current / INTENTLOG_DIR).is_dir():
            return current
        current = current.parent

    # Check root
    if (current / INTENTLOG_DIR).is_dir():
        return current

    return None


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
        """Get the path to a branch's intent file"""
        if branch_name == "main":
            return self.intentlog_dir / INTENTS_FILE
        return self.branches_dir / f"{branch_name}.json"

    def _lock_file(self, file_path: Path, exclusive: bool = True):
        """
        Context manager for file locking.

        Uses fcntl for Unix file locking to prevent concurrent writes.
        """
        class FileLock:
            def __init__(self, path: Path, exclusive: bool):
                self.path = path
                self.exclusive = exclusive
                self.file = None

            def __enter__(self):
                self.file = open(self.path, 'a+')
                lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
                fcntl.flock(self.file.fileno(), lock_type)
                return self.file

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.file:
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                    self.file.close()
                return False

        return FileLock(file_path, exclusive)

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
        """
        if self.is_initialized() and not force:
            raise ProjectExistsError(
                f"IntentLog already initialized in {self.project_root}"
            )

        # Create directory structure
        self.intentlog_dir.mkdir(exist_ok=True)
        self.branches_dir.mkdir(exist_ok=True)

        # Create config
        config = ProjectConfig(project_name=project_name)
        with open(self.config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Create empty main branch intents
        main_intents_path = self._get_branch_file("main")
        with open(main_intents_path, 'w') as f:
            json.dump({"intents": [], "branch": "main"}, f, indent=2)

        # Create .gitignore for sensitive data
        gitignore_path = self.intentlog_dir / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write("# IntentLog sensitive data\n")
            f.write("*.key\n")
            f.write("*.secret\n")
            f.write("temp/\n")

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
                f"No IntentLog project found. Run 'ilog init' first."
            )

        with open(self.config_path, 'r') as f:
            data = json.load(f)

        return ProjectConfig.from_dict(data)

    def save_config(self, config: ProjectConfig) -> None:
        """Save project configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

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
            with open(branch_file, 'r') as f:
                data = json.load(f)

        intents = []
        for intent_data in data.get("intents", []):
            intent = Intent(
                intent_id=intent_data["intent_id"],
                intent_name=intent_data["intent_name"],
                intent_reasoning=intent_data["intent_reasoning"],
                timestamp=datetime.fromisoformat(intent_data["timestamp"]),
                metadata=intent_data.get("metadata", {}),
                parent_intent_id=intent_data.get("parent_intent_id"),
            )
            intents.append(intent)

        return intents

    def save_intents(self, intents: List[Intent], branch: Optional[str] = None) -> None:
        """
        Save intents to a branch.

        Args:
            intents: List of Intent objects to save
            branch: Branch name. If None, uses current branch.
        """
        config = self.load_config()
        branch = branch or config.current_branch

        branch_file = self._get_branch_file(branch)

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

        with self._lock_file(branch_file, exclusive=True):
            with open(branch_file, 'w') as f:
                json.dump(data, f, indent=2)

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
        intent = Intent(
            intent_name=name,
            intent_reasoning=reasoning,
            metadata=metadata or {},
            parent_intent_id=parent_id,
        )

        if not intent.validate():
            raise ValueError("Intent must have name and non-empty reasoning")

        intents = self.load_intents(branch)
        intents.append(intent)
        self.save_intents(intents, branch)

        return intent

    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> None:
        """
        Create a new branch.

        Args:
            branch_name: Name for the new branch
            from_branch: Branch to copy from. If None, uses current branch.

        Raises:
            BranchExistsError: If branch already exists
        """
        config = self.load_config()
        from_branch = from_branch or config.current_branch

        new_branch_file = self._get_branch_file(branch_name)
        if new_branch_file.is_file():
            raise BranchExistsError(f"Branch '{branch_name}' already exists")

        # Copy intents from source branch
        intents = self.load_intents(from_branch)

        # Create new branch file
        data = {
            "branch": branch_name,
            "created_from": from_branch,
            "created_at": datetime.now().isoformat(),
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

        with open(new_branch_file, 'w') as f:
            json.dump(data, f, indent=2)

    def switch_branch(self, branch_name: str) -> None:
        """
        Switch to a different branch.

        Args:
            branch_name: Branch to switch to

        Raises:
            BranchNotFoundError: If branch doesn't exist
        """
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
            with open(branch_file, 'r') as f:
                data = json.load(f)

        chained = []
        for intent_data in data.get("intents", []):
            # Check if chain data exists
            if "chain_hash" in intent_data:
                # Load with existing chain data
                chained.append(ChainedIntent.from_dict(intent_data))
            else:
                # Legacy data without chain - need to rebuild
                intent = Intent(
                    intent_id=intent_data["intent_id"],
                    intent_name=intent_data["intent_name"],
                    intent_reasoning=intent_data["intent_reasoning"],
                    timestamp=datetime.fromisoformat(intent_data["timestamp"]),
                    metadata=intent_data.get("metadata", {}),
                    parent_intent_id=intent_data.get("parent_intent_id"),
                )
                chained.append(intent)

        # If any items are plain Intent, rebuild the chain
        if chained and isinstance(chained[0], Intent):
            plain_intents = [c if isinstance(c, Intent) else c.intent for c in chained]
            chained = chain_intents(plain_intents)

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
            with open(branch_file, 'w') as f:
                json.dump(data, f, indent=2)

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

        # Load existing chain
        chained_intents = self.load_chained_intents(branch)

        # Append to chain
        new_chained = append_to_chain(chained_intents, intent)

        # Sign if requested
        if sign:
            signature = self._sign_chained_intent(new_chained, key_password)
            new_chained.signature = signature

        chained_intents.append(new_chained)
        self.save_chained_intents(chained_intents, branch)

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
        from .merkle import compute_root_hash, GENESIS_HASH

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

    def migrate_to_chain(self, branch: Optional[str] = None) -> int:
        """
        Migrate legacy intents to chain format.

        Args:
            branch: Branch to migrate. If None, uses current branch.

        Returns:
            Number of intents migrated
        """
        from .merkle import chain_intents

        # Load plain intents
        intents = self.load_intents(branch)
        if not intents:
            return 0

        # Chain them
        chained = chain_intents(intents)

        # Save with chain data
        self.save_chained_intents(chained, branch)

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

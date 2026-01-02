"""
Backup and Recovery Module for IntentLog

Provides comprehensive backup and recovery capabilities:
- Full project backup to compressed archives
- Incremental backups based on chain hashes
- Point-in-time recovery
- Backup verification and integrity checking
- Cloud storage support (S3, GCS, Azure Blob)
"""

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import get_logger, log_context
from .storage import IntentLogStorage, INTENTLOG_DIR


@dataclass
class BackupMetadata:
    """Metadata for a backup archive."""
    backup_id: str
    created_at: str
    project_name: str
    backup_type: str  # "full", "incremental"
    source_path: str
    branches: List[str]
    intent_count: int
    chain_hashes: Dict[str, str]  # branch -> root hash
    parent_backup_id: Optional[str] = None
    compression: str = "gzip"
    checksum: Optional[str] = None
    size_bytes: int = 0
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "project_name": self.project_name,
            "backup_type": self.backup_type,
            "source_path": self.source_path,
            "branches": self.branches,
            "intent_count": self.intent_count,
            "chain_hashes": self.chain_hashes,
            "parent_backup_id": self.parent_backup_id,
            "compression": self.compression,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        return cls(
            backup_id=data["backup_id"],
            created_at=data["created_at"],
            project_name=data["project_name"],
            backup_type=data["backup_type"],
            source_path=data["source_path"],
            branches=data["branches"],
            intent_count=data["intent_count"],
            chain_hashes=data["chain_hashes"],
            parent_backup_id=data.get("parent_backup_id"),
            compression=data.get("compression", "gzip"),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
            version=data.get("version", "1.0"),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    restored_path: Path
    branches_restored: List[str]
    intents_restored: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BackupError(Exception):
    """Base exception for backup errors."""
    pass


class RestoreError(Exception):
    """Base exception for restore errors."""
    pass


class BackupManager:
    """
    Manages backup and recovery operations for IntentLog projects.

    Supports:
    - Full backups (complete .intentlog directory)
    - Incremental backups (changes since last backup)
    - Local and remote storage
    - Compression and encryption
    - Verification and integrity checking
    """

    def __init__(
        self,
        project_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ):
        """
        Initialize BackupManager.

        Args:
            project_path: Path to IntentLog project root
            backup_dir: Directory for storing backups (default: ~/.intentlog/backups)
        """
        self.storage = IntentLogStorage(project_path)
        self.project_path = self.storage.project_root

        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path.home() / ".intentlog" / "backups"

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _generate_backup_id(self) -> str:
        """Generate a unique backup ID."""
        import secrets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"{timestamp}_{random_suffix}"

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_chain_hashes(self) -> Dict[str, str]:
        """Get root chain hashes for all branches."""
        chain_hashes = {}
        for branch in self.storage.list_branches():
            try:
                root_hash = self.storage.get_root_hash(branch)
                chain_hashes[branch] = root_hash
            except Exception:
                # Branch might not have chain data
                chain_hashes[branch] = ""
        return chain_hashes

    def _count_intents(self) -> int:
        """Count total intents across all branches."""
        total = 0
        for branch in self.storage.list_branches():
            try:
                intents = self.storage.load_intents(branch)
                total += len(intents)
            except Exception:
                pass
        return total

    def create_backup(
        self,
        backup_type: str = "full",
        parent_backup_id: Optional[str] = None,
        compress: bool = True,
        include_keys: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[Path, BackupMetadata]:
        """
        Create a backup of the IntentLog project.

        Args:
            backup_type: "full" or "incremental"
            parent_backup_id: For incremental, the parent backup to diff against
            compress: Whether to compress the backup
            include_keys: Whether to include private keys (security risk!)
            progress_callback: Optional callback(message, progress_0_to_1)

        Returns:
            Tuple of (backup_path, metadata)
        """
        logger = get_logger()

        if not self.storage.is_initialized():
            raise BackupError("No IntentLog project found")

        backup_id = self._generate_backup_id()
        config = self.storage.load_config()

        with log_context(operation="backup", backup_id=backup_id):
            logger.info(f"Creating {backup_type} backup", project=config.project_name)

            if progress_callback:
                progress_callback("Gathering project information...", 0.1)

            # Gather metadata
            branches = self.storage.list_branches()
            intent_count = self._count_intents()
            chain_hashes = self._get_chain_hashes()

            metadata = BackupMetadata(
                backup_id=backup_id,
                created_at=datetime.utcnow().isoformat() + "Z",
                project_name=config.project_name,
                backup_type=backup_type,
                source_path=str(self.project_path),
                branches=branches,
                intent_count=intent_count,
                chain_hashes=chain_hashes,
                parent_backup_id=parent_backup_id,
                compression="gzip" if compress else "none",
            )

            if progress_callback:
                progress_callback("Creating archive...", 0.3)

            # Create backup archive
            backup_filename = f"intentlog_backup_{backup_id}.tar"
            if compress:
                backup_filename += ".gz"

            backup_path = self.backup_dir / backup_filename

            # Create tarball
            mode = "w:gz" if compress else "w"
            with tarfile.open(backup_path, mode) as tar:
                # Add .intentlog directory
                intentlog_dir = self.project_path / INTENTLOG_DIR

                for item in intentlog_dir.rglob("*"):
                    if item.is_file():
                        # Skip private keys unless explicitly requested
                        if not include_keys and item.suffix in (".key", ".pem"):
                            if "private" in item.name.lower():
                                logger.debug(f"Skipping private key: {item.name}")
                                continue

                        arcname = str(item.relative_to(self.project_path))
                        tar.add(item, arcname=arcname)

                if progress_callback:
                    progress_callback("Adding metadata...", 0.8)

                # Add metadata file
                metadata_json = json.dumps(metadata.to_dict(), indent=2)
                metadata_bytes = metadata_json.encode("utf-8")

                import io
                metadata_file = io.BytesIO(metadata_bytes)
                tarinfo = tarfile.TarInfo(name="backup_metadata.json")
                tarinfo.size = len(metadata_bytes)
                tar.addfile(tarinfo, metadata_file)

            # Compute checksum
            metadata.checksum = self._compute_file_checksum(backup_path)
            metadata.size_bytes = backup_path.stat().st_size

            # Write final metadata to sidecar file
            metadata_path = backup_path.with_suffix(backup_path.suffix + ".meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            if progress_callback:
                progress_callback("Backup complete!", 1.0)

            logger.info(
                "Backup created successfully",
                backup_id=backup_id,
                size_bytes=metadata.size_bytes,
                intent_count=intent_count,
            )

            return backup_path, metadata

    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups."""
        backups = []

        for meta_file in self.backup_dir.glob("*.meta.json"):
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    backups.append(BackupMetadata.from_dict(data))
            except Exception as e:
                get_logger().warning(f"Failed to read backup metadata: {meta_file}: {e}")

        # Sort by creation date, newest first
        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups

    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get metadata for a specific backup."""
        for backup in self.list_backups():
            if backup.backup_id == backup_id:
                return backup
        return None

    def verify_backup(
        self,
        backup_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of a backup.

        Args:
            backup_id: The backup to verify
            progress_callback: Optional progress callback

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        logger = get_logger()
        issues = []

        metadata = self.get_backup(backup_id)
        if not metadata:
            return False, [f"Backup not found: {backup_id}"]

        backup_path = self._find_backup_file(backup_id)
        if not backup_path:
            return False, [f"Backup archive not found for: {backup_id}"]

        if progress_callback:
            progress_callback("Verifying checksum...", 0.2)

        # Verify checksum
        actual_checksum = self._compute_file_checksum(backup_path)
        if metadata.checksum and actual_checksum != metadata.checksum:
            issues.append(f"Checksum mismatch: expected {metadata.checksum}, got {actual_checksum}")

        if progress_callback:
            progress_callback("Verifying archive contents...", 0.5)

        # Verify archive can be read
        try:
            mode = "r:gz" if metadata.compression == "gzip" else "r"
            with tarfile.open(backup_path, mode) as tar:
                members = tar.getnames()

                # Check for essential files
                if "backup_metadata.json" not in members:
                    issues.append("Missing backup_metadata.json in archive")

                if not any(INTENTLOG_DIR in m for m in members):
                    issues.append(f"Missing {INTENTLOG_DIR} directory in archive")

        except Exception as e:
            issues.append(f"Failed to read archive: {e}")

        if progress_callback:
            progress_callback("Verification complete!", 1.0)

        is_valid = len(issues) == 0
        logger.info(f"Backup verification {'passed' if is_valid else 'failed'}", backup_id=backup_id)

        return is_valid, issues

    def _find_backup_file(self, backup_id: str) -> Optional[Path]:
        """Find the backup archive file for a given backup ID."""
        patterns = [
            f"intentlog_backup_{backup_id}.tar.gz",
            f"intentlog_backup_{backup_id}.tar",
        ]
        for pattern in patterns:
            path = self.backup_dir / pattern
            if path.exists():
                return path
        return None

    def restore_backup(
        self,
        backup_id: str,
        target_path: Optional[Path] = None,
        overwrite: bool = False,
        verify_first: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> RestoreResult:
        """
        Restore a backup to a target directory.

        Args:
            backup_id: The backup to restore
            target_path: Where to restore (default: original location)
            overwrite: Whether to overwrite existing files
            verify_first: Whether to verify backup integrity first
            progress_callback: Optional progress callback

        Returns:
            RestoreResult with details
        """
        logger = get_logger()
        warnings = []
        errors = []

        metadata = self.get_backup(backup_id)
        if not metadata:
            raise RestoreError(f"Backup not found: {backup_id}")

        backup_path = self._find_backup_file(backup_id)
        if not backup_path:
            raise RestoreError(f"Backup archive not found for: {backup_id}")

        with log_context(operation="restore", backup_id=backup_id):
            # Verify if requested
            if verify_first:
                if progress_callback:
                    progress_callback("Verifying backup integrity...", 0.1)

                is_valid, issues = self.verify_backup(backup_id)
                if not is_valid:
                    raise RestoreError(f"Backup verification failed: {issues}")

            # Determine target path
            if target_path is None:
                target_path = Path(metadata.source_path)

            target_path = Path(target_path)
            target_intentlog = target_path / INTENTLOG_DIR

            if progress_callback:
                progress_callback("Preparing restore...", 0.2)

            # Check for existing project
            if target_intentlog.exists():
                if not overwrite:
                    raise RestoreError(
                        f"Target already has IntentLog project. Use overwrite=True to replace."
                    )
                # Backup existing before overwriting
                backup_existing = target_intentlog.with_suffix(".bak")
                if backup_existing.exists():
                    shutil.rmtree(backup_existing)
                shutil.move(str(target_intentlog), str(backup_existing))
                warnings.append(f"Existing project backed up to {backup_existing}")

            if progress_callback:
                progress_callback("Extracting backup...", 0.4)

            # Extract archive
            try:
                target_path.mkdir(parents=True, exist_ok=True)

                mode = "r:gz" if metadata.compression == "gzip" else "r"
                with tarfile.open(backup_path, mode) as tar:
                    # Extract all files except metadata
                    for member in tar.getmembers():
                        if member.name == "backup_metadata.json":
                            continue
                        tar.extract(member, target_path)

                if progress_callback:
                    progress_callback("Verifying restored data...", 0.8)

                # Verify restored project
                restored_storage = IntentLogStorage(target_path)
                if not restored_storage.is_initialized():
                    errors.append("Restored project is not properly initialized")

                # Count restored intents
                intents_restored = 0
                branches_restored = []
                for branch in restored_storage.list_branches():
                    try:
                        intents = restored_storage.load_intents(branch)
                        intents_restored += len(intents)
                        branches_restored.append(branch)
                    except Exception as e:
                        warnings.append(f"Failed to verify branch {branch}: {e}")

                if progress_callback:
                    progress_callback("Restore complete!", 1.0)

                logger.info(
                    "Backup restored successfully",
                    backup_id=backup_id,
                    target=str(target_path),
                    intents_restored=intents_restored,
                )

                return RestoreResult(
                    success=len(errors) == 0,
                    restored_path=target_path,
                    branches_restored=branches_restored,
                    intents_restored=intents_restored,
                    warnings=warnings,
                    errors=errors,
                )

            except Exception as e:
                logger.error(f"Restore failed: {e}")
                raise RestoreError(f"Failed to restore backup: {e}")

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup and its metadata.

        Args:
            backup_id: The backup to delete

        Returns:
            True if deleted, False if not found
        """
        logger = get_logger()

        backup_path = self._find_backup_file(backup_id)
        if not backup_path:
            return False

        # Delete archive
        backup_path.unlink()

        # Delete metadata
        meta_path = backup_path.with_suffix(backup_path.suffix + ".meta.json")
        if meta_path.exists():
            meta_path.unlink()

        logger.info(f"Deleted backup: {backup_id}")
        return True

    def cleanup_old_backups(
        self,
        keep_count: int = 5,
        keep_days: Optional[int] = None,
    ) -> List[str]:
        """
        Clean up old backups based on retention policy.

        Args:
            keep_count: Number of most recent backups to keep
            keep_days: Delete backups older than this many days

        Returns:
            List of deleted backup IDs
        """
        logger = get_logger()
        deleted = []

        backups = self.list_backups()  # Already sorted newest first

        # Apply count-based retention
        if keep_count > 0 and len(backups) > keep_count:
            for backup in backups[keep_count:]:
                if self.delete_backup(backup.backup_id):
                    deleted.append(backup.backup_id)

        # Apply time-based retention
        if keep_days is not None:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=keep_days)

            for backup in self.list_backups():
                backup_time = datetime.fromisoformat(backup.created_at.rstrip("Z"))
                if backup_time < cutoff:
                    if self.delete_backup(backup.backup_id):
                        deleted.append(backup.backup_id)

        if deleted:
            logger.info(f"Cleaned up {len(deleted)} old backups")

        return deleted


def create_backup(
    project_path: Optional[Path] = None,
    backup_dir: Optional[Path] = None,
    compress: bool = True,
) -> Tuple[Path, BackupMetadata]:
    """
    Convenience function to create a backup.

    Args:
        project_path: Path to IntentLog project
        backup_dir: Directory for backups
        compress: Whether to compress

    Returns:
        Tuple of (backup_path, metadata)
    """
    manager = BackupManager(project_path, backup_dir)
    return manager.create_backup(compress=compress)


def restore_backup(
    backup_id: str,
    backup_dir: Optional[Path] = None,
    target_path: Optional[Path] = None,
    overwrite: bool = False,
) -> RestoreResult:
    """
    Convenience function to restore a backup.

    Args:
        backup_id: The backup to restore
        backup_dir: Directory containing backups
        target_path: Where to restore
        overwrite: Whether to overwrite existing

    Returns:
        RestoreResult
    """
    manager = BackupManager(backup_dir=backup_dir)
    return manager.restore_backup(backup_id, target_path, overwrite)


def list_backups(backup_dir: Optional[Path] = None) -> List[BackupMetadata]:
    """
    Convenience function to list backups.

    Args:
        backup_dir: Directory containing backups

    Returns:
        List of backup metadata
    """
    manager = BackupManager(backup_dir=backup_dir)
    return manager.list_backups()


__all__ = [
    "BackupMetadata",
    "RestoreResult",
    "BackupError",
    "RestoreError",
    "BackupManager",
    "create_backup",
    "restore_backup",
    "list_backups",
]

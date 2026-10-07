"""
End-to-end tests for the backup and recovery module.

These tests run BackupManager against a real project on disk and need no
optional dependencies (no cryptography, no cloud SDKs).
"""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from intentlog.backup import (
    BackupError,
    BackupManager,
    BackupMetadata,
    RestoreError,
    create_backup,
    list_backups,
    restore_backup,
)
from intentlog.logging import LogContext, get_current_log_context, log_context
from intentlog.storage import CONFIG_FILE, INTENTLOG_DIR, INTENTS_FILE, IntentLogStorage

INTENT_FIXTURES = [
    ("choose storage format", "JSON keeps the core dependency-free"),
    ("add hash chaining", "tamper evidence without a database"),
    ("ship backups", "operators need point-in-time recovery"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intent_tuples(project_dir: Path, branch=None):
    """Stable, comparable view of the intents stored in a project."""
    storage = IntentLogStorage(project_dir)
    return [
        (i.intent_id, i.intent_name, i.intent_reasoning, i.timestamp, i.metadata)
        for i in storage.load_intents(branch)
    ]


@pytest.fixture
def project(tmp_path):
    """An initialized project with a few intents on main."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = IntentLogStorage(project_dir)
    storage.init_project("backup-test")
    for name, reasoning in INTENT_FIXTURES:
        storage.add_intent(name, reasoning, metadata={"source": "test"})
    return project_dir


@pytest.fixture
def backup_dir(tmp_path):
    return tmp_path / "backups"


@pytest.fixture
def manager(project, backup_dir):
    return BackupManager(project, backup_dir=backup_dir)


# ---- Logging context (root cause of the create/restore crash) ----


class TestLogContextExtraKwargs:
    """log_context() must accept ad-hoc fields instead of raising TypeError."""

    def test_unknown_kwargs_are_routed_into_extra(self):
        with log_context(operation="backup", backup_id="abc123") as ctx:
            assert ctx.operation == "backup"
            assert ctx.extra == {"backup_id": "abc123"}
            assert ctx.to_dict() == {"operation": "backup", "backup_id": "abc123"}
            assert get_current_log_context() is ctx

    def test_explicit_extra_merges_with_unknown_kwargs(self):
        with log_context(extra={"a": 1}, b=2) as ctx:
            assert ctx.extra == {"a": 1, "b": 2}

    def test_nested_contexts_merge_and_restore(self):
        with log_context(operation="restore", backup_id="outer") as outer:
            with log_context(target="/tmp/x") as inner:
                assert inner.operation == "restore"
                assert inner.extra == {"backup_id": "outer", "target": "/tmp/x"}
            assert get_current_log_context() is outer
            assert outer.extra == {"backup_id": "outer"}
        assert get_current_log_context().to_dict() == {}

    def test_from_kwargs_keeps_known_fields_as_fields(self):
        ctx = LogContext.from_kwargs(intent_id="i-1", branch="main", foo="bar")
        assert ctx.intent_id == "i-1"
        assert ctx.branch == "main"
        assert ctx.extra == {"foo": "bar"}
        assert ctx.to_dict() == {"intent_id": "i-1", "branch": "main", "foo": "bar"}

    def test_from_kwargs_with_no_arguments(self):
        assert LogContext.from_kwargs() == LogContext()


# ---- Create ----


class TestCreateBackup:
    def test_create_backup_writes_archive_and_sidecar(self, manager, backup_dir):
        backup_path, metadata = manager.create_backup()

        assert backup_path.parent == backup_dir
        assert backup_path.name == f"intentlog_backup_{metadata.backup_id}.tar.gz"
        assert backup_path.is_file()

        sidecar = backup_dir / f"{backup_path.name}.meta.json"
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text()) == metadata.to_dict()

        assert isinstance(metadata, BackupMetadata)
        assert metadata.project_name == "backup-test"
        assert metadata.backup_type == "full"
        assert metadata.compression == "gzip"
        assert metadata.branches == ["main"]
        assert metadata.intent_count == len(INTENT_FIXTURES)
        assert "main" in metadata.chain_hashes
        assert metadata.checksum == _sha256(backup_path)
        assert metadata.size_bytes == backup_path.stat().st_size

    def test_archive_contains_project_files_and_metadata(self, manager):
        backup_path, _ = manager.create_backup()

        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            metadata_member = tar.extractfile("backup_metadata.json")
            assert metadata_member is not None
            embedded = json.loads(metadata_member.read().decode("utf-8"))

        assert f"{INTENTLOG_DIR}/{CONFIG_FILE}" in names
        assert f"{INTENTLOG_DIR}/{INTENTS_FILE}" in names
        assert embedded["project_name"] == "backup-test"

    def test_create_backup_uncompressed(self, manager):
        backup_path, metadata = manager.create_backup(compress=False)

        assert backup_path.suffix == ".tar"
        assert metadata.compression == "none"
        assert manager.verify_backup(metadata.backup_id) == (True, [])

    def test_private_keys_skipped_unless_requested(self, manager, project):
        keys_dir = project / INTENTLOG_DIR
        (keys_dir / "private_key.pem").write_text("secret")
        (keys_dir / "public_key.pem").write_text("public")

        backup_path, _ = manager.create_backup()
        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
        assert f"{INTENTLOG_DIR}/private_key.pem" not in names
        assert f"{INTENTLOG_DIR}/public_key.pem" in names

        backup_path, _ = manager.create_backup(include_keys=True)
        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
        assert f"{INTENTLOG_DIR}/private_key.pem" in names

    def test_create_backup_requires_initialized_project(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        manager = BackupManager(empty, backup_dir=tmp_path / "backups")

        with pytest.raises(BackupError):
            manager.create_backup()

    def test_progress_callback_is_invoked(self, manager):
        calls = []
        manager.create_backup(progress_callback=lambda msg, p: calls.append((msg, p)))

        assert calls
        assert all(0.0 <= p <= 1.0 for _, p in calls)
        assert calls[-1][1] == 1.0

    def test_create_backup_leaves_log_context_untouched(self, manager):
        assert get_current_log_context().to_dict() == {}
        manager.create_backup()
        assert get_current_log_context().to_dict() == {}


# ---- List / verify ----


class TestListAndVerify:
    def test_list_backups_newest_first(self, manager):
        _, first = manager.create_backup()
        _, second = manager.create_backup()

        listed = manager.list_backups()
        assert [b.backup_id for b in listed] == [second.backup_id, first.backup_id]

    def test_get_backup(self, manager):
        _, metadata = manager.create_backup()

        found = manager.get_backup(metadata.backup_id)
        assert found is not None
        assert found.to_dict() == metadata.to_dict()
        assert manager.get_backup("does-not-exist") is None

    def test_verify_backup_passes(self, manager):
        _, metadata = manager.create_backup()
        assert manager.verify_backup(metadata.backup_id) == (True, [])

    def test_verify_backup_unknown_id(self, manager):
        is_valid, issues = manager.verify_backup("missing")
        assert is_valid is False
        assert issues == ["Backup not found: missing"]

    def test_verify_backup_detects_checksum_mismatch(self, manager):
        backup_path, metadata = manager.create_backup()
        with open(backup_path, "ab") as f:
            f.write(b"tampered")

        is_valid, issues = manager.verify_backup(metadata.backup_id)
        assert is_valid is False
        assert any("Checksum mismatch" in issue for issue in issues)

    def test_verify_backup_missing_archive(self, manager):
        backup_path, metadata = manager.create_backup()
        backup_path.unlink()

        is_valid, issues = manager.verify_backup(metadata.backup_id)
        assert is_valid is False
        assert issues == [f"Backup archive not found for: {metadata.backup_id}"]


# ---- Restore ----


class TestRestoreBackup:
    def test_restore_into_fresh_directory(self, manager, project, tmp_path):
        _, metadata = manager.create_backup()
        target = tmp_path / "restored"

        result = manager.restore_backup(metadata.backup_id, target_path=target)

        assert result.success is True
        assert result.errors == []
        assert result.warnings == []
        assert result.restored_path == target
        assert result.branches_restored == ["main"]
        assert result.intents_restored == len(INTENT_FIXTURES)

        restored = IntentLogStorage(target)
        assert restored.is_initialized()
        assert restored.load_config().project_name == "backup-test"
        assert _intent_tuples(target) == _intent_tuples(project)
        assert not (target / "backup_metadata.json").exists()

    def test_restore_uncompressed_backup(self, manager, project, tmp_path):
        _, metadata = manager.create_backup(compress=False)
        target = tmp_path / "restored"

        result = manager.restore_backup(metadata.backup_id, target_path=target)

        assert result.success is True
        assert _intent_tuples(target) == _intent_tuples(project)

    def test_restore_refuses_existing_project_without_overwrite(self, manager, project):
        _, metadata = manager.create_backup()

        with pytest.raises(RestoreError, match="overwrite"):
            manager.restore_backup(metadata.backup_id, target_path=project)

    def test_restore_with_overwrite_keeps_previous_copy(self, manager, project):
        _, metadata = manager.create_backup()
        IntentLogStorage(project).add_intent("later", "added after the backup")
        assert len(_intent_tuples(project)) == len(INTENT_FIXTURES) + 1

        result = manager.restore_backup(
            metadata.backup_id, target_path=project, overwrite=True
        )

        assert result.success is True
        assert result.intents_restored == len(INTENT_FIXTURES)
        assert len(_intent_tuples(project)) == len(INTENT_FIXTURES)
        assert any(".bak" in w for w in result.warnings)
        previous = project / f"{INTENTLOG_DIR}.bak"
        assert (previous / CONFIG_FILE).is_file()
        assert len(json.loads((previous / INTENTS_FILE).read_text())["intents"]) == 4

    def test_restore_unknown_backup_raises(self, manager, tmp_path):
        with pytest.raises(RestoreError, match="Backup not found"):
            manager.restore_backup("missing", target_path=tmp_path / "x")

    def test_restore_rejects_tampered_archive(self, manager, tmp_path):
        backup_path, metadata = manager.create_backup()
        with open(backup_path, "ab") as f:
            f.write(b"tampered")

        with pytest.raises(RestoreError, match="verification failed"):
            manager.restore_backup(metadata.backup_id, target_path=tmp_path / "x")

    def test_restore_leaves_log_context_untouched(self, manager, tmp_path):
        _, metadata = manager.create_backup()
        manager.restore_backup(metadata.backup_id, target_path=tmp_path / "restored")
        assert get_current_log_context().to_dict() == {}


# ---- Delete / cleanup ----


class TestDeleteAndCleanup:
    def test_delete_backup_removes_archive_and_sidecar(self, manager, backup_dir):
        _, metadata = manager.create_backup()

        assert manager.delete_backup(metadata.backup_id, confirm=True) is True
        assert list(backup_dir.iterdir()) == []
        assert manager.delete_backup(metadata.backup_id, confirm=True) is False

    def test_cleanup_old_backups_keeps_most_recent(self, manager):
        _, oldest = manager.create_backup()
        _, middle = manager.create_backup()
        _, newest = manager.create_backup()

        deleted = manager.cleanup_old_backups(keep_count=2)

        assert deleted == [oldest.backup_id]
        remaining = {b.backup_id for b in manager.list_backups()}
        assert remaining == {middle.backup_id, newest.backup_id}


# ---- Module-level convenience functions ----


class TestConvenienceFunctions:
    def test_create_list_restore_roundtrip(self, project, backup_dir, tmp_path, monkeypatch):
        # The restore/list helpers resolve the project from cwd; keep it away
        # from any real project on the developer's machine.
        monkeypatch.chdir(tmp_path)

        backup_path, metadata = create_backup(project, backup_dir)
        assert backup_path.is_file()

        listed = list_backups(backup_dir)
        assert [b.backup_id for b in listed] == [metadata.backup_id]

        target = tmp_path / "restored"
        result = restore_backup(metadata.backup_id, backup_dir=backup_dir, target_path=target)
        assert result.success is True
        assert _intent_tuples(target) == _intent_tuples(project)

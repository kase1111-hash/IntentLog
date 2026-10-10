"""
Tests for IntentLog storage module
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pytest

from intentlog.core import Intent
from intentlog.filelock import LOCK_BACKEND, FileLock, FileLockError, FileLockTimeout
from intentlog.storage import (
    INTENTLOG_DIR,
    BranchExistsError,
    BranchNotFoundError,
    IntentLogStorage,
    ProjectConfig,
    ProjectExistsError,
    ProjectNotFoundError,
    compute_intent_hash,
    find_project_root,
)
from intentlog.validation import ValidationError


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield Path(temp_dir)
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


@pytest.fixture
def initialized_project(temp_project_dir):
    """Create and initialize a project"""
    storage = IntentLogStorage(temp_project_dir)
    storage.init_project("test-project")
    return storage


class TestProjectConfig:
    """Tests for ProjectConfig dataclass"""

    def test_config_creation(self):
        """Test creating a config"""
        config = ProjectConfig(project_name="my-project")
        assert config.project_name == "my-project"
        assert config.current_branch == "main"
        assert config.version == "0.2.0"

    def test_config_to_dict(self):
        """Test serializing config to dict"""
        config = ProjectConfig(project_name="my-project")
        data = config.to_dict()
        assert data["project_name"] == "my-project"
        assert data["current_branch"] == "main"
        assert "created_at" in data

    def test_config_from_dict(self):
        """Test deserializing config from dict"""
        data = {
            "project_name": "my-project",
            "created_at": "2025-01-01T00:00:00",
            "current_branch": "develop",
            "version": "0.2.0",
        }
        config = ProjectConfig.from_dict(data)
        assert config.project_name == "my-project"
        assert config.current_branch == "develop"
        assert config.version == "0.2.0"


class TestComputeIntentHash:
    """Tests for intent hash computation"""

    def test_hash_is_deterministic(self):
        """Test that same intent produces same hash"""
        intent = Intent(
            intent_id="test-id",
            intent_name="Test Intent",
            intent_reasoning="This is a test",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )
        hash1 = compute_intent_hash(intent)
        hash2 = compute_intent_hash(intent)
        assert hash1 == hash2

    def test_different_intents_different_hashes(self):
        """Test that different intents produce different hashes"""
        intent1 = Intent(
            intent_id="id-1",
            intent_name="Intent 1",
            intent_reasoning="First intent",
        )
        intent2 = Intent(
            intent_id="id-2",
            intent_name="Intent 2",
            intent_reasoning="Second intent",
        )
        assert compute_intent_hash(intent1) != compute_intent_hash(intent2)

    def test_hash_is_short(self):
        """Test that hash is truncated to 12 characters"""
        intent = Intent(
            intent_name="Test",
            intent_reasoning="Test reasoning",
        )
        hash_val = compute_intent_hash(intent)
        assert len(hash_val) == 12


class TestIntentLogStorage:
    """Tests for IntentLogStorage class"""

    def test_init_project(self, temp_project_dir):
        """Test initializing a new project"""
        storage = IntentLogStorage(temp_project_dir)
        config = storage.init_project("my-project")

        assert config.project_name == "my-project"
        assert storage.intentlog_dir.is_dir()
        assert storage.config_path.is_file()
        assert storage.branches_dir.is_dir()

    def test_init_project_creates_gitignore(self, temp_project_dir):
        """Test that init creates .gitignore"""
        storage = IntentLogStorage(temp_project_dir)
        storage.init_project("my-project")

        gitignore = storage.intentlog_dir / ".gitignore"
        assert gitignore.is_file()

    def test_init_project_already_exists(self, initialized_project):
        """Test that init fails if project already exists"""
        with pytest.raises(ProjectExistsError):
            initialized_project.init_project("another-project")

    def test_init_project_force(self, initialized_project):
        """Test that init with force reinitializes"""
        config = initialized_project.init_project("new-name", force=True)
        assert config.project_name == "new-name"

    def test_is_initialized(self, temp_project_dir):
        """Test checking if project is initialized"""
        storage = IntentLogStorage(temp_project_dir)
        assert not storage.is_initialized()

        storage.init_project("test")
        assert storage.is_initialized()

    def test_load_config(self, initialized_project):
        """Test loading project config"""
        config = initialized_project.load_config()
        assert config.project_name == "test-project"
        assert config.current_branch == "main"

    def test_load_config_not_initialized(self, temp_project_dir):
        """Test loading config fails if not initialized"""
        storage = IntentLogStorage(temp_project_dir)
        with pytest.raises(ProjectNotFoundError):
            storage.load_config()

    def test_add_intent(self, initialized_project):
        """Test adding an intent"""
        intent = initialized_project.add_intent(
            name="Test Intent",
            reasoning="This is a test reasoning",
        )
        assert intent.intent_name == "Test Intent"
        assert intent.intent_reasoning == "This is a test reasoning"

    def test_add_intent_with_metadata(self, initialized_project):
        """Test adding an intent with metadata"""
        intent = initialized_project.add_intent(
            name="Test Intent",
            reasoning="This is a test",
            metadata={"key": "value"},
        )
        assert intent.metadata["key"] == "value"

    def test_add_intent_invalid(self, initialized_project):
        """Test adding invalid intent fails"""
        with pytest.raises(ValueError):
            initialized_project.add_intent(
                name="",
                reasoning="Empty name",
            )

    def test_load_intents(self, initialized_project):
        """Test loading intents"""
        initialized_project.add_intent("Intent 1", "Reasoning 1")
        initialized_project.add_intent("Intent 2", "Reasoning 2")

        intents = initialized_project.load_intents()
        assert len(intents) == 2
        assert intents[0].intent_name == "Intent 1"
        assert intents[1].intent_name == "Intent 2"

    def test_save_intents(self, initialized_project):
        """Test saving intents"""
        intent = Intent(
            intent_name="Saved Intent",
            intent_reasoning="Saved reasoning",
        )
        initialized_project.save_intents([intent])

        loaded = initialized_project.load_intents()
        assert len(loaded) == 1
        assert loaded[0].intent_name == "Saved Intent"


class TestBranching:
    """Tests for branch operations"""

    def test_list_branches_default(self, initialized_project):
        """Test listing branches shows main by default"""
        branches = initialized_project.list_branches()
        assert "main" in branches

    def test_create_branch(self, initialized_project):
        """Test creating a new branch"""
        initialized_project.create_branch("feature-x")
        branches = initialized_project.list_branches()
        assert "feature-x" in branches

    def test_create_branch_copies_intents(self, initialized_project):
        """Test that new branch copies intents from current"""
        initialized_project.add_intent("Original", "Original reasoning")
        initialized_project.create_branch("feature-x")

        intents = initialized_project.load_intents("feature-x")
        assert len(intents) == 1
        assert intents[0].intent_name == "Original"

    def test_create_branch_already_exists(self, initialized_project):
        """Test creating existing branch fails"""
        initialized_project.create_branch("feature-x")
        with pytest.raises(BranchExistsError):
            initialized_project.create_branch("feature-x")

    def test_switch_branch(self, initialized_project):
        """Test switching branches"""
        initialized_project.create_branch("develop")
        initialized_project.switch_branch("develop")

        config = initialized_project.load_config()
        assert config.current_branch == "develop"

    def test_switch_branch_not_found(self, initialized_project):
        """Test switching to non-existent branch fails"""
        with pytest.raises(BranchNotFoundError):
            initialized_project.switch_branch("nonexistent")

    def test_branch_isolation(self, initialized_project):
        """Test that changes on one branch don't affect another"""
        # Add intent on main
        initialized_project.add_intent("Main Intent", "On main branch")

        # Create and switch to feature branch
        initialized_project.create_branch("feature")
        initialized_project.switch_branch("feature")

        # Add intent on feature
        initialized_project.add_intent("Feature Intent", "On feature branch")

        # Check main still has only one intent
        main_intents = initialized_project.load_intents("main")
        assert len(main_intents) == 1
        assert main_intents[0].intent_name == "Main Intent"

        # Check feature has two intents
        feature_intents = initialized_project.load_intents("feature")
        assert len(feature_intents) == 2


class TestSearch:
    """Tests for search functionality"""

    def test_search_by_name(self, initialized_project):
        """Test searching intents by name"""
        initialized_project.add_intent("Architecture Decision", "We chose microservices")
        initialized_project.add_intent("Bug Fix", "Fixed the login issue")

        results = initialized_project.search_intents("Architecture")
        assert len(results) == 1
        assert results[0].intent_name == "Architecture Decision"

    def test_search_by_reasoning(self, initialized_project):
        """Test searching intents by reasoning content"""
        initialized_project.add_intent("Design Choice", "We chose microservices for scalability")
        initialized_project.add_intent("Bug Fix", "Fixed the login issue")

        results = initialized_project.search_intents("microservices")
        assert len(results) == 1
        assert "microservices" in results[0].intent_reasoning

    def test_search_case_insensitive(self, initialized_project):
        """Test that search is case insensitive"""
        initialized_project.add_intent("Test Intent", "UPPERCASE reasoning")

        results = initialized_project.search_intents("uppercase")
        assert len(results) == 1

    def test_search_no_results(self, initialized_project):
        """Test search with no matches"""
        initialized_project.add_intent("Test Intent", "Some reasoning")

        results = initialized_project.search_intents("nonexistent")
        assert len(results) == 0


class TestFindProjectRoot:
    """Tests for find_project_root function"""

    def test_find_in_current_dir(self, temp_project_dir):
        """Test finding project root in current directory"""
        storage = IntentLogStorage(temp_project_dir)
        storage.init_project("test")

        found = find_project_root(temp_project_dir)
        assert found == temp_project_dir

    def test_find_in_parent_dir(self, temp_project_dir):
        """Test finding project root in parent directory"""
        storage = IntentLogStorage(temp_project_dir)
        storage.init_project("test")

        # Create and cd to subdirectory
        subdir = temp_project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        found = find_project_root(subdir)
        assert found == temp_project_dir

    def test_not_found(self, temp_project_dir):
        """Test when no project root exists"""
        found = find_project_root(temp_project_dir)
        assert found is None


def _try_lock_in_thread(path, exclusive=True, timeout=0.3):
    """Try to take a FileLock on `path` from another thread; return True if acquired."""
    outcome = {}

    def worker():
        try:
            with FileLock(path, exclusive=exclusive, timeout=timeout):
                outcome["acquired"] = True
        except FileLockTimeout:
            outcome["acquired"] = False

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout + 5)
    assert not t.is_alive(), "lock attempt in helper thread hung"
    return outcome["acquired"]


class TestFileLockReentrancy:
    """Tests for the per-thread re-entrant FileLock"""

    def test_nested_locks_in_same_thread_do_not_block(self, temp_project_dir):
        """Nested exclusive/shared requests inside an exclusive lock are no-ops"""
        path = temp_project_dir / "data.json.lock"
        outer = FileLock(path, exclusive=True, timeout=0.5)
        with outer:
            with FileLock(path, exclusive=True, timeout=0.5) as inner:
                assert inner.is_locked
                with FileLock(path, exclusive=False, timeout=0.5):
                    pass
            # Releasing the nested locks must not drop the outer lock
            assert outer.is_locked
            assert _try_lock_in_thread(path) is False
        assert not outer.is_locked
        assert _try_lock_in_thread(path) is True

    def test_other_threads_are_excluded(self, temp_project_dir):
        """A lock held by one thread blocks exclusive requests from another"""
        path = temp_project_dir / "data.json.lock"
        with FileLock(path, exclusive=True, timeout=0.5):
            assert _try_lock_in_thread(path, exclusive=True) is False
            assert _try_lock_in_thread(path, exclusive=False) is False
        assert _try_lock_in_thread(path, exclusive=True) is True

    @pytest.mark.skipif(LOCK_BACKEND != "fcntl", reason="shared locks need fcntl")
    def test_shared_locks_allow_concurrent_readers(self, temp_project_dir):
        """Shared locks from different threads do not exclude each other"""
        path = temp_project_dir / "data.json.lock"
        with FileLock(path, exclusive=False, timeout=0.5):
            assert _try_lock_in_thread(path, exclusive=False) is True
            assert _try_lock_in_thread(path, exclusive=True) is False

    def test_shared_to_exclusive_upgrade_is_rejected(self, temp_project_dir):
        """Upgrading a held shared lock to exclusive fails fast, not by timeout"""
        path = temp_project_dir / "data.json.lock"
        with FileLock(path, exclusive=False, timeout=0.5):
            with pytest.raises(FileLockError):
                FileLock(path, exclusive=True, timeout=0.5).acquire()

    def test_release_without_acquire_is_noop(self, temp_project_dir):
        """Releasing a lock that was never acquired does nothing"""
        lock = FileLock(temp_project_dir / "data.json.lock")
        lock.release()
        assert not lock.is_locked


class TestConcurrentAccess:
    """Regression tests for lost updates under concurrent writes"""

    def test_concurrent_add_intent_persists_every_write(self, initialized_project):
        """8 threads x 25 add_intent calls -> exactly 200 unique intents persisted"""
        project_root = initialized_project.project_root
        num_threads = 8
        per_thread = 25
        errors = []
        barrier = threading.Barrier(num_threads)

        def worker(idx):
            storage = IntentLogStorage(project_root)
            try:
                barrier.wait(timeout=5)
                for i in range(per_thread):
                    storage.add_intent(
                        name=f"Thread {idx} intent {i}",
                        reasoning=f"Concurrent write {idx}-{i}",
                    )
            except Exception as e:  # pragma: no cover - reported via assert
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "writer threads hung"
        assert errors == []

        intents = initialized_project.load_intents()
        assert len(intents) == num_threads * per_thread
        assert len({i.intent_id for i in intents}) == num_threads * per_thread
        names = {i.intent_name for i in intents}
        assert len(names) == num_threads * per_thread

    def test_concurrent_add_chained_intent_keeps_chain_valid(self, initialized_project):
        """Concurrent chained writes persist everything and leave a valid chain"""
        project_root = initialized_project.project_root
        num_threads = 4
        per_thread = 10
        errors = []

        def worker(idx):
            storage = IntentLogStorage(project_root)
            try:
                for i in range(per_thread):
                    storage.add_chained_intent(
                        name=f"Chained {idx}-{i}",
                        reasoning="Concurrent chained write",
                    )
            except Exception as e:  # pragma: no cover - reported via assert
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == []
        chained = initialized_project.load_chained_intents()
        assert len(chained) == num_threads * per_thread
        assert initialized_project.verify_chain().valid

    def test_nested_operations_inside_lock_branch_do_not_deadlock(self, initialized_project):
        """load/save/add inside lock_branch() reuse the held lock instead of blocking"""
        storage = initialized_project
        storage.lock_timeout = 1.0

        with storage.lock_branch() as branch:
            assert branch == "main"
            assert storage.load_intents() == []
            storage.save_intents([Intent(intent_name="Saved", intent_reasoning="inside lock")])
            storage.add_intent("Added", "inside lock")
            with storage.lock_branch("main"):
                storage.add_chained_intent("Chained", "inside nested lock")
            assert len(storage.load_chained_intents()) == 3
            # Still held: another thread must not get the exclusive lock
            lock_path = storage._get_lock_file(storage._get_branch_file("main"))
            assert _try_lock_in_thread(lock_path) is False

        # Released on exit: another thread can now take it
        assert _try_lock_in_thread(lock_path) is True
        assert [i.intent_name for i in storage.load_intents()] == ["Saved", "Added", "Chained"]

    def test_lock_branch_excludes_other_threads(self, initialized_project):
        """lock_branch() held in one thread blocks lock_branch() in another"""
        storage = initialized_project
        other = IntentLogStorage(storage.project_root)
        other.lock_timeout = 0.3
        outcome = {}

        def try_lock():
            try:
                with other.lock_branch("main"):
                    outcome["result"] = "acquired"
            except FileLockTimeout:
                outcome["result"] = "timeout"

        with storage.lock_branch("main"):
            t = threading.Thread(target=try_lock)
            t.start()
            t.join(timeout=5)
        assert outcome["result"] == "timeout"

        t = threading.Thread(target=try_lock)
        t.start()
        t.join(timeout=5)
        assert outcome["result"] == "acquired"

    def test_lock_branch_resolves_current_branch(self, initialized_project):
        """lock_branch(None) locks and yields the current branch"""
        storage = initialized_project
        storage.create_branch("develop")
        storage.switch_branch("develop")
        with storage.lock_branch() as branch:
            assert branch == "develop"
            storage.add_intent("On develop", "inside lock")
        assert len(storage.load_intents("develop")) == 1
        assert storage.load_intents("main") == []

    @pytest.mark.skipif(LOCK_BACKEND == "none", reason="fallback backend removes its lock file")
    def test_lock_sidecar_is_not_listed_as_branch(self, initialized_project):
        """Locks live on '<file>.lock' sidecars that list_branches() ignores"""
        storage = initialized_project
        storage.create_branch("feature")
        storage.add_intent("On feature", "sidecar test", branch="feature")
        storage.add_intent("On main", "sidecar test")

        assert (storage.branches_dir / "feature.json.lock").is_file()
        assert (storage.intentlog_dir / "intents.json.lock").is_file()
        assert storage.list_branches() == ["feature", "main"]
        # The data file itself is still the only thing read back
        with open(storage.branches_dir / "feature.json") as f:
            assert len(json.load(f)["intents"]) == 1

    def test_gitignore_excludes_lock_files(self, initialized_project):
        """init writes '*.lock' into .intentlog/.gitignore"""
        gitignore = initialized_project.intentlog_dir / ".gitignore"
        assert "*.lock" in gitignore.read_text().splitlines()


class TestLockBranchSideEffects:
    """lock_branch() must validate before it creates any lock sidecar"""

    @staticmethod
    def _failing_writes():
        """Write paths that route through lock_branch() with an explicit branch"""
        return [
            ("add_intent", lambda s, b: s.add_intent("n", "r", branch=b)),
            ("add_chained_intent", lambda s, b: s.add_chained_intent("n", "r", branch=b)),
            ("migrate_to_chain", lambda s, b: s.migrate_to_chain(b)),
            ("lock_branch", lambda s, b: s.lock_branch(b).__enter__()),
        ]

    def test_uninitialized_dir_raises_without_creating_intentlog(self, temp_project_dir):
        """A failed write on an uninitialized dir leaves no .intentlog/ behind"""
        for label, call in self._failing_writes():
            target = temp_project_dir / label
            target.mkdir()
            storage = IntentLogStorage(target)
            with pytest.raises(ProjectNotFoundError):
                call(storage, "main")
            assert not (target / INTENTLOG_DIR).exists(), label
            assert list(target.iterdir()) == [], label
            assert not storage.is_initialized()

    def test_failed_write_in_subdir_does_not_shadow_project_root(self, initialized_project):
        """A stray .intentlog/ in a subdirectory would hijack find_project_root()"""
        root = initialized_project.project_root
        sub = root / "src" / "pkg"
        sub.mkdir(parents=True)
        assert find_project_root(sub) == root

        with pytest.raises(ProjectNotFoundError):
            IntentLogStorage(sub).add_intent("n", "r", branch="main")

        assert not (sub / INTENTLOG_DIR).exists()
        assert find_project_root(sub) == root

    def test_traversal_branch_name_raises_without_creating_files(self, initialized_project):
        """A traversal-style branch name is rejected before any lock file is created"""
        storage = initialized_project
        root = storage.project_root
        before = set(root.rglob("*"))

        for label, call in self._failing_writes():
            with pytest.raises(ValidationError):
                call(storage, "../../outside")
            with pytest.raises(ValidationError):
                call(storage, "../../../escaped")
            assert set(root.rglob("*")) == before, label

        assert not (root / "outside.json.lock").exists()
        assert not (root.parent / "escaped.json.lock").exists()
        assert storage.list_branches() == ["main"]

    def test_lock_branch_accepts_valid_names(self, initialized_project):
        """Validation does not reject ordinary branch names, including main"""
        storage = initialized_project
        storage.create_branch("feature-x.1_y")
        with storage.lock_branch("main") as branch:
            assert branch == "main"
        with storage.lock_branch("feature-x.1_y") as branch:
            assert branch == "feature-x.1_y"
            storage.add_intent("n", "r", branch=branch)
        assert len(storage.load_intents("feature-x.1_y")) == 1

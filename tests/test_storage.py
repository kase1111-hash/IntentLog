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

from intentlog import CRYPTO_AVAILABLE
from intentlog.core import Intent
from intentlog.filelock import LOCK_BACKEND, FileLock, FileLockError, FileLockTimeout
from intentlog.storage import (
    INTENTLOG_DIR,
    BranchExistsError,
    BranchNotFoundError,
    ChainIntegrityError,
    IntentLogStorage,
    ProjectConfig,
    ProjectExistsError,
    ProjectNotFoundError,
    SignatureLossError,
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
    def test_lock_files_live_under_temp(self, initialized_project):
        """Locks live in .intentlog/temp/locks/, never next to the data files"""
        storage = initialized_project
        storage.create_branch("feature")
        storage.add_intent("On feature", "lock file test", branch="feature")
        storage.add_intent("On main", "lock file test")

        locks = storage.intentlog_dir / "temp" / "locks"
        assert (locks / "branches" / "feature.json.lock").is_file()
        assert (locks / "intents.json.lock").is_file()
        assert not list(storage.intentlog_dir.glob("*.lock"))
        assert not list(storage.branches_dir.glob("*.lock"))
        assert storage.list_branches() == ["feature", "main"]
        # The data file itself is still the only thing read back
        with open(storage.branches_dir / "feature.json") as f:
            assert len(json.load(f)["intents"]) == 1

    def test_gitignore_excludes_lock_files(self, initialized_project):
        """Lock files sit under temp/, which init has always written to .gitignore"""
        gitignore = initialized_project.intentlog_dir / ".gitignore"
        assert "temp/" in gitignore.read_text().splitlines()

    def test_config_branch_name_is_validated_before_locking(self, initialized_project):
        """A traversal-style current_branch in config.json never creates files"""
        storage = initialized_project
        config_file = storage.intentlog_dir / "config.json"
        config = json.loads(config_file.read_text())
        config["current_branch"] = "../../escape"
        config_file.write_text(json.dumps(config))
        project_root = storage.intentlog_dir.parent

        with pytest.raises(ValidationError):
            storage.add_intent("a", "b")

        assert not list(project_root.rglob("escape*"))
        assert not list(project_root.parent.glob("escape*"))


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


# =========================================================================
# Chain data and signatures survive ordinary writes
# =========================================================================


def _branch_data(storage, branch="main"):
    with open(storage._get_branch_file(branch)) as f:
        return json.load(f)


def _tamper(storage, index, branch="main"):
    """Rewrite one stored intent's reasoning without updating its hashes."""
    path = storage._get_branch_file(branch)
    data = json.loads(path.read_text())
    data["intents"][index]["intent_reasoning"] = "rewritten after the fact"
    path.write_text(json.dumps(data, indent=2))


class TestChainPreservation:
    """save_intents/create_branch/migrate_to_chain keep the stored chain."""

    def test_plain_branch_stays_plain(self, initialized_project):
        storage = initialized_project
        storage.add_intent("a", "plain one")
        storage.add_intent("b", "plain two")

        data = _branch_data(storage)
        assert "chain_version" not in data
        assert all("chain_hash" not in e for e in data["intents"])
        assert all("hash" in e for e in data["intents"])

    def test_append_keeps_existing_chain_entries(self, initialized_project):
        storage = initialized_project
        storage.add_chained_intent("a", "first")
        storage.add_chained_intent("b", "second")
        before = _branch_data(storage)["intents"]

        storage.add_intent("c", "plain append")

        after = _branch_data(storage)["intents"]
        assert after[:2] == before
        assert after[2]["prev_hash"] == before[1]["chain_hash"]
        assert after[2]["sequence"] == 2
        assert storage.verify_chain().valid

    def test_edit_rechains_only_from_changed_intent(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        before = _branch_data(storage)["intents"]

        intents = storage.load_intents()
        intents[1].metadata["tags"] = ["x"]
        assert storage.save_intents(intents, strict=True) == 0

        after = _branch_data(storage)["intents"]
        assert after[0] == before[0]
        assert after[1]["metadata"]["tags"] == ["x"]
        assert after[1]["chain_hash"] != before[1]["chain_hash"]
        assert after[2]["prev_hash"] == after[1]["chain_hash"]
        assert storage.verify_chain().valid

    def test_unchanged_save_rewrites_nothing(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        before = _branch_data(storage)

        storage.save_intents(storage.load_intents(), strict=True)

        assert _branch_data(storage) == before

    def test_create_branch_copies_chain(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")

        storage.create_branch("feat")

        feat = _branch_data(storage, "feat")
        assert feat["intents"] == _branch_data(storage)["intents"]
        assert feat["created_from"] == "main"
        assert feat["root_hash"] == storage.get_root_hash("main")
        assert storage.verify_chain("feat").valid

    def test_migrate_keeps_existing_chain(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        before = _branch_data(storage)

        assert storage.migrate_to_chain() == 2
        assert _branch_data(storage) == before

    def test_migrate_chains_plain_branch(self, initialized_project):
        storage = initialized_project
        storage.add_intent("a", "plain one")
        storage.add_intent("b", "plain two")

        assert storage.migrate_to_chain() == 2

        data = _branch_data(storage)
        assert all("chain_hash" in e for e in data["intents"])
        assert storage.verify_chain().valid

    def test_strict_rewrite_over_broken_entry_is_refused(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        _tamper(storage, 1)
        before = _branch_data(storage)

        intents = storage.load_intents()
        intents[0].metadata["tags"] = ["x"]
        with pytest.raises(ChainIntegrityError):
            storage.save_intents(intents, strict=True)

        assert _branch_data(storage) == before
        assert not storage.verify_chain().valid

    def test_edit_after_broken_entry_keeps_the_evidence(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        _tamper(storage, 0)

        intents = storage.load_intents()
        intents[2].metadata["tags"] = ["x"]
        storage.save_intents(intents, strict=True)

        data = _branch_data(storage)
        assert data["intents"][0]["intent_reasoning"] == "rewritten after the fact"
        assert data["intents"][2]["metadata"]["tags"] == ["x"]
        assert not storage.verify_chain().valid

    def test_append_to_broken_chain_keeps_the_evidence(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        _tamper(storage, 0)

        storage.add_intent("c", "appended after tampering")

        assert not storage.verify_chain().valid


class TestStrippedChainEntries:
    """Entries whose chain data was removed by hand are treated as damage."""

    @staticmethod
    def _strip_last(storage):
        path = storage._get_branch_file("main")
        data = json.loads(path.read_text())
        for key in ("intent_hash", "prev_hash", "chain_hash", "sequence", "signature"):
            data["intents"][-1].pop(key, None)
        data["intents"][-1]["intent_reasoning"] = "rewritten after the fact"
        path.write_text(json.dumps(data, indent=2))

    def test_verify_reports_instead_of_crashing(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._strip_last(storage)

        result = storage.verify_chain()

        assert not result.valid
        assert result.broken_at == 1

    def test_strict_rewrite_of_stripped_entry_is_refused(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._strip_last(storage)
        before = _branch_data(storage)

        intents = storage.load_intents()
        intents[1].metadata["tags"] = ["x"]
        with pytest.raises(ChainIntegrityError):
            storage.save_intents(intents, strict=True)

        assert _branch_data(storage) == before

    def test_migrate_refuses_unless_forced(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._strip_last(storage)

        with pytest.raises(ChainIntegrityError):
            storage.migrate_to_chain()
        assert not storage.verify_chain().valid

        assert storage.migrate_to_chain(force=True) == 2
        assert storage.verify_chain().valid


def _strip(storage, index, branch="main"):
    path = storage._get_branch_file(branch)
    data = json.loads(path.read_text())
    for key in ("intent_hash", "prev_hash", "chain_hash", "sequence", "signature"):
        data["intents"][index].pop(key, None)
    path.write_text(json.dumps(data, indent=2))


class TestStrippedFirstEntry:
    """Stripping the first entry must not turn the branch back into 'legacy'."""

    def test_verify_reports_stripped_first_entry(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        _strip(storage, 0)
        _tamper(storage, 2)

        result = storage.verify_chain()

        assert not result.valid
        assert result.broken_at == 0

    def test_commit_on_stripped_branch_is_refused(self, initialized_project):
        storage = initialized_project
        for name in "ab":
            storage.add_chained_intent(name, f"reasoning {name}")
        _strip(storage, 0)
        before = _branch_data(storage)

        with pytest.raises(ChainIntegrityError, match="migrate --force"):
            storage.add_chained_intent("c", "appended to a damaged branch")

        assert _branch_data(storage) == before

    def test_migrate_force_repairs_stripped_and_empty_entries(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        path = storage._get_branch_file("main")
        data = json.loads(path.read_text())
        data["intents"][1]["chain_hash"] = ""  # as an older build could persist
        path.write_text(json.dumps(data, indent=2))
        _strip(storage, 2)

        assert storage.migrate_to_chain(force=True) == 3

        assert storage.verify_chain().valid
        assert all(e["chain_hash"] for e in _branch_data(storage)["intents"])

    def test_branch_copy_of_damaged_chain_stays_repairable(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        _strip(storage, 1)

        storage.create_branch("copy")

        assert _branch_data(storage, "copy")["intents"] == _branch_data(storage)["intents"]
        assert not storage.verify_chain("copy").valid
        storage.migrate_to_chain("copy", force=True)
        assert storage.verify_chain("copy").valid


class TestUnreadableChainedEntry:
    """A chained entry that cannot be parsed is reported, never dropped."""

    @staticmethod
    def _break_entry(storage, index):
        path = storage._get_branch_file("main")
        data = json.loads(path.read_text())
        del data["intents"][index]["prev_hash"]
        path.write_text(json.dumps(data, indent=2))

    def test_verify_reports_unreadable_entry(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._break_entry(storage, 1)

        result = storage.verify_chain()

        assert not result.valid
        assert result.entries_checked == 3

    def test_commit_does_not_drop_unreadable_entry(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._break_entry(storage, 1)
        before = _branch_data(storage)

        with pytest.raises(ChainIntegrityError):
            storage.add_chained_intent("d", "appended to a damaged branch")

        assert _branch_data(storage) == before


class TestNonDictAndUnparseableEntries:
    """Any unreadable entry on a chained branch is damage, wherever it is."""

    @staticmethod
    def _set_entry(storage, index, value):
        path = storage._get_branch_file("main")
        data = json.loads(path.read_text())
        data["intents"][index] = value
        path.write_text(json.dumps(data, indent=2))

    @pytest.mark.parametrize("value", [None, "garbage", 42, ["x"]])
    def test_non_dict_tail_entry_is_reported_and_not_dropped(self, initialized_project, value):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        self._set_entry(storage, 2, value)
        before = _branch_data(storage)

        assert not storage.verify_chain().valid
        assert len(storage.load_intents()) == 2  # readable entries, no crash
        with pytest.raises(ChainIntegrityError):
            storage.add_chained_intent("d", "appended to a damaged branch")
        intents = storage.load_intents()
        intents[0].metadata["tags"] = ["x"]
        with pytest.raises(ChainIntegrityError):
            storage.save_intents(intents, strict=True)

        assert _branch_data(storage) == before

    def test_stripped_and_unparseable_entry_is_reported(self, initialized_project):
        storage = initialized_project
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}")
        _strip(storage, 2)
        path = storage._get_branch_file("main")
        data = json.loads(path.read_text())
        data["intents"][2]["timestamp"] = "yesterday"
        path.write_text(json.dumps(data, indent=2))

        assert not storage.verify_chain().valid
        with pytest.raises(ChainIntegrityError):
            storage.add_chained_intent("d", "appended to a damaged branch")


class TestBranchNamesOnEveryPath:
    @pytest.mark.parametrize("name", ["../escape", "../../x", "a/b", "bad name"])
    def test_invalid_names_are_not_found_and_create_nothing(self, initialized_project, name):
        storage = initialized_project
        root = storage.intentlog_dir.parent
        before = sorted(str(p) for p in root.parent.glob("*"))

        with pytest.raises(BranchNotFoundError):
            storage.load_intents(name)
        with pytest.raises(BranchNotFoundError):
            storage.save_intents([], name)

        assert sorted(str(p) for p in root.parent.glob("*")) == before
        assert not list((storage.intentlog_dir / "temp").rglob("*escape*"))


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
class TestSignedChainPreservation:
    """Signatures survive every write that does not change their intents."""

    @pytest.fixture
    def signed_project(self, initialized_project):
        storage = initialized_project
        with pytest.warns(UserWarning):
            storage.get_key_manager().generate_key()
        for name in "abc":
            storage.add_chained_intent(name, f"reasoning {name}", sign=True)
        return storage

    @staticmethod
    def _signed(storage, branch="main"):
        return [e.get("signature") is not None for e in _branch_data(storage, branch)["intents"]]

    def test_strict_edit_is_refused_and_leaves_file_unchanged(self, signed_project):
        storage = signed_project
        before = _branch_data(storage)
        intents = storage.load_intents()
        intents[0].metadata["tags"] = ["x"]

        with pytest.raises(SignatureLossError) as excinfo:
            storage.save_intents(intents, strict=True)

        assert excinfo.value.count == 3
        assert excinfo.value.branch == "main"
        assert _branch_data(storage) == before

    def test_edit_removes_only_invalidated_signatures(self, signed_project):
        storage = signed_project
        intents = storage.load_intents()
        intents[2].metadata["tags"] = ["x"]

        assert storage.save_intents(intents) == 1

        assert self._signed(storage) == [True, True, False]
        chained = storage.load_chained_intents()
        assert storage.verify_intent_signature(chained[0])
        assert storage.verify_intent_signature(chained[1])
        assert storage.verify_chain().valid

    def test_append_keeps_signatures_valid(self, signed_project):
        storage = signed_project
        storage.add_intent("d", "plain append")

        assert self._signed(storage) == [True, True, True, False]
        for chained in storage.load_chained_intents()[:3]:
            assert storage.verify_intent_signature(chained)

    def test_migrate_keeps_signatures(self, signed_project):
        storage = signed_project
        before = _branch_data(storage)

        assert storage.migrate_to_chain() == 3

        assert _branch_data(storage) == before
        assert self._signed(storage) == [True, True, True]

    def test_create_branch_keeps_signatures_valid(self, signed_project):
        storage = signed_project
        storage.create_branch("feat")

        assert self._signed(storage, "feat") == [True, True, True]
        for chained in storage.load_chained_intents("feat"):
            assert storage.verify_intent_signature(chained)


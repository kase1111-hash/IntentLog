"""
Tests for IntentLog storage module
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest

from intentlog.storage import (
    IntentLogStorage,
    ProjectConfig,
    ProjectNotFoundError,
    ProjectExistsError,
    BranchNotFoundError,
    BranchExistsError,
    compute_intent_hash,
    find_project_root,
    INTENTLOG_DIR,
)
from intentlog.core import Intent


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
        assert config.version == "0.1.0"

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

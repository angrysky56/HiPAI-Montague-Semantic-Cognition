import pytest

from hipai.synthesis import HIPAIManager


@pytest.fixture
def manager():
    """Provides a HIPAIManager instance with an in-memory ontology."""
    # Use :memory: by default for tests to avoid database locking issues
    mgr = HIPAIManager(db_path=":memory:")
    # Clear the graph to ensure test isolation
    mgr.world_model.clear_graph()
    yield mgr
    mgr.close()


@pytest.fixture
def disk_manager(tmp_path):
    """
    Provides a HIPAIManager instance with a disk-based ontology in a temp directory.
    """
    db_file = tmp_path / "test_world.db"
    mgr = HIPAIManager(db_path=str(db_file))
    yield mgr
    mgr.close()

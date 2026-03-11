from unittest.mock import MagicMock, patch

import pytest

from hipai.models import Individual, Observation, Relation
from hipai.world_model import WorldModel


@pytest.fixture
def mock_falkordb():
    with patch("hipai.world_model.FalkorDB") as mock:
        yield mock


def test_incorporate_observation(mock_falkordb):
    # Setup mock
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph
    mock_falkordb.return_value = mock_db

    world_model = WorldModel()

    # reset mock because WorldModel.__init__ calls `_ensure_graph` which makes queries
    mock_graph.reset_mock()

    obs = Observation(
        text_source="Dog is an animal",
        individuals=[
            Individual(id="e1", name="Dog", properties=["Furry"]),
            Individual(id="e2", name="Animal"),
        ],
        relations=[Relation(source_id="e1", target_id="e2", relation_type="IS_A")],
    )

    world_model.incorporate_observation(obs)

    # Check that graph.query was called 6 times:
    # 1 base for Observation, 3 for e1 (merge + prop check + prop set), 1 for e2, 1 for relation
    assert mock_graph.query.call_count == 6


def test_query_graph(mock_falkordb):
    mock_db = MagicMock()
    mock_graph = MagicMock()

    mock_result = MagicMock()
    mock_result.result_set = [{"name": "Dog"}]
    mock_graph.query.return_value = mock_result

    mock_db.select_graph.return_value = mock_graph
    mock_falkordb.return_value = mock_db

    world_model = WorldModel()
    mock_graph.reset_mock()

    result = world_model.query_graph("MATCH (n) RETURN n")

    assert result == [{"name": "Dog"}]
    mock_graph.query.assert_called_with("MATCH (n) RETURN n", params={})

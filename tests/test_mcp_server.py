import pytest
import json
from hipai.mcp_server import add_belief, hi_pai

@pytest.fixture(autouse=True)
def setup_graph():
    hi_pai.world_model.clear_graph()

@pytest.mark.asyncio
async def test_mcp_add_belief_success():
    """Test successful belief addition via MCP."""
    res_str = await add_belief("Socrates is a man")
    res = json.loads(res_str)
    assert res["status"] == "success"

@pytest.mark.asyncio
async def test_mcp_ambiguity_response():
    """Test that MCP server returns the structured AmbiguityDetected string."""
    # This should trigger multiple patterns as seen in test_ambiguity.py
    res = await add_belief("Socrates is a teacher who causes trouble")
    
    assert "AmbiguityDetected" in res
    assert "Option 1" in res
    assert "Option 2" in res
    assert "Please call the tool again" in res

@pytest.mark.asyncio
async def test_mcp_pruning_resolution():
    """Test that MCP server resolves ambiguity if pruning leaves only one valid parse."""
    # Seed axiom: Socrates cannot HARM Students
    from hipai.models import DeontologicalAxiom
    hi_pai.world_model.incorporate_axiom(DeontologicalAxiom(
        tier="T1",
        subject_type="Socrates",
        relation_type="HARM",
        object_type="Student",
        constraint="FORBIDDEN",
        source_axiom="A1"
    ))
    hi_pai.world_model.query_graph("MERGE (p:Entity {id: 'Plato'}) MERGE (c:Concept {name: 'Concept_Student'}) MERGE (p)-[:INSTANCE_OF]->(c)")
    
    # Statement: 'Socrates is a teacher who harms Plato'
    # Parse 1: Property (Valid)
    # Parse 2: Relation HARM (Invalid)
    # Should result in success, not AmbiguityDetected
    res_str = await add_belief("Socrates is a teacher who harms Plato")
    res = json.loads(res_str)
    assert res["status"] == "success"

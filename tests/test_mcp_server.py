import json

import pytest

from hipai.mcp_server import add_belief, hi_pai
from hipai.models import DeontologicalAxiom


@pytest.fixture(autouse=True)
def setup_isolated_db(tmp_path):
    """Ensure hi_pai uses an isolated database for each test."""
    # Since hi_pai is global, we clear it.
    hi_pai.world_model.clear_database()
    yield


@pytest.mark.asyncio
async def test_mcp_add_belief_success():
    """Test successful belief addition via MCP."""
    res_str = await add_belief("Socrates is a man")
    res = json.loads(res_str)
    assert res["status"] == "success"
    # The new parser lowercases IDs
    assert res["observation"]["individuals"][0]["id"] == "socrates"


@pytest.mark.asyncio
async def test_mcp_entity_ambiguity():
    """Test that MCP server handles entity resolution (Alice -> Alice Smith)."""
    # 1. Add Alice Smith
    await add_belief("Alice Smith is a human")

    # 2. Add Alice is happy
    res_str = await add_belief("Alice is happy")
    res = json.loads(res_str)

    assert res["status"] == "success"
    # Should have resolved Alice to alice_smith
    assert res["observation"]["individuals"][0]["id"] == "alice_smith"


@pytest.mark.asyncio
async def test_mcp_constraint_violation():
    """Test that MCP server returns deontological violations."""
    hi_pai.incorporate_axiom(
        DeontologicalAxiom(
            tier="T1",
            subject_type="Agent",
            relation_type="HARMS",
            object_type="Patient",
            constraint="FORBIDDEN",
            source_axiom="A1",
        )
    )
    await add_belief("Alice is a patient")

    res_str = await add_belief("Agent harms Alice")
    res = json.loads(res_str)

    assert res["status"] == "error"
    assert "Action blocked" in res["message"]

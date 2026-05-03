import pytest
from hipai.synthesis import HIPAIManager
from hipai.models import DeontologicalAxiom



def test_unambiguous_belief(manager):
    """Test that a simple, unambiguous belief is added successfully."""
    res = manager.add_belief("Socrates is a man")
    assert res["status"] == "success"
    
    # Verify in graph
    state = manager.get_current_state()
    # Note: IDs are lowercased and underscores in the new parser
    assert any(n["properties"].get("id") == "socrates" for n in state["nodes"])

def test_entity_ambiguity_resolution(manager):
    """
    Test that 'Alice' is correctly mapped to 'Alice Smith' if Alice Smith already exists.
    This verifies the semantic search hardening.
    """
    # 1. Add 'Alice Smith'
    manager.add_belief("Alice Smith is a human")
    
    # 2. Add belief about 'Alice'
    res = manager.add_belief("Alice is happy")
    assert res["status"] == "success"
    
    # 3. Verify that 'Alice' was mapped to 'alice_smith'
    # In the new logic, the observation returned should have the resolved ID
    obs = res["observation"]
    assert obs.individuals[0].id == "alice_smith"
    
    # Verify in graph there is only one Alice-related node
    state = manager.get_current_state()
    entity_nodes = [n for n in state["nodes"] if n["label"] in ["Entity", "ContentNode"]]
    # Should only have alice_smith, not a separate 'alice'
    ids = [n["properties"].get("id") for n in entity_nodes]
    assert "alice_smith" in ids
    assert "alice" not in ids

def test_pruning_with_axioms(manager):
    """
    Test that invalid relations are blocked by T1 constraints.
    """
    # 1. Seed the axiom
    manager.incorporate_axiom(DeontologicalAxiom(
        tier="T1",
        subject_type="Agent",
        relation_type="HARMS",
        object_type="Human",
        constraint="FORBIDDEN",
        source_axiom="A1"
    ))
    
    # 2. Add an entity 'Alice' who is a Patient
    manager.add_belief("Alice is a patient")
    
    # 3. Try to add a belief that violates the axiom
    res = manager.add_belief("Agent harms Alice")
    
    assert res["status"] == "error"
    assert "Deontological violation" in res["message"]

def test_recursive_ambiguity_resolution(manager):
    """
    Test that entities in nested observations are also resolved.
    """
    # 1. Add 'Bob Jones'
    manager.add_belief("Bob Jones is a teacher")
    import time
    time.sleep(1) # Wait for vector index sync
    
    # 2. Add recursive belief about 'Bob'
    # 'Alice believes Bob is happy'
    res = manager.add_belief("Alice believes Bob is happy")
    assert res["status"] == "success"
    
    # 3. Verify 'Bob' resolved to 'bob_jones' in the nested observation
    rel = res["observation"].relations[0]
    inner_obs = rel.target_observation
    assert inner_obs.individuals[0].id == "bob_jones"

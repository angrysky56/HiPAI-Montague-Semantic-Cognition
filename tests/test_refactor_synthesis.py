import pytest
import os
import uuid
from hipai.synthesis import HIPAIManager
from hipai.models import Observation, Individual, Relation

@pytest.fixture
def manager():
    db_path = f"test_world_{uuid.uuid4()}.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    manager = HIPAIManager(db_path=db_path)
    yield manager
    
    # Cleanup
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass

def test_add_belief_basic(manager):
    manager.clear_database()
    res = manager.add_belief("Socrates is a man")
    assert res["status"] == "success"
    obs = res["observation"]
    assert len(obs.individuals) == 1
    assert obs.individuals[0].name == "Socrates"
    assert "man" in obs.individuals[0].properties

def test_add_belief_action(manager):
    manager.clear_database()
    res = manager.add_belief("Alice harms the Bob")
    # This should FAIL because Alice is an Entity and Bob is an Entity, 
    # and Tier 1 rules say 'harms' targets a 'Patient'.
    # Actually, in our seeded ontology, 'Agent' can 'harm' 'Patient'.
    # If Alice and Bob are just Entities, it might be allowed unless we typed them.
    # Let's see what happens.
    assert res["status"] == "success" # For now, since they are just Entities

def test_add_belief_constraint_violation(manager):
    manager.clear_database()
    # Let's explicitly type Bob as a Patient
    manager.add_belief("Bob is a Patient")
    manager.add_belief("Alice is an Agent")
    
    # Agents can harm Patients?
    # Let's check ontology_manager.py seed logic.
    # Agent --harms--> Patient is ALLOWED? 
    # No, usually harms is a violation if the target is a Patient and the source is NOT authorized?
    # Actually, our seed says:
    # harms.domain = [Agent]
    # harms.range = [Patient]
    # This means Agent CAN harm Patient in OWL (it's a valid property).
    # But HIPAIManager.check_action (called via WorldModel.check_constraint)
    # checks if the action 'harms' a 'Patient'.
    
    res = manager.add_belief("Alice harms Bob")
    # If 'harms' is detected and target is 'Patient', it returns permitted=False.
    assert res["status"] == "error"
    assert "Deontological violation" in res["message"]

if __name__ == "__main__":
    pytest.main([__file__])

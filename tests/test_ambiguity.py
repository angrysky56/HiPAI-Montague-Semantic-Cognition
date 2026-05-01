import pytest
from hipai.synthesis import HIPAIManager
from hipai.exceptions import AmbiguityDetectedError
from hipai.models import DeontologicalAxiom

@pytest.fixture
def manager():
    manager = HIPAIManager(graph_name="test_ambiguity")
    manager.world_model.clear_graph()
    return manager

def test_unambiguous_belief(manager):
    """Test that a simple, unambiguous belief is added successfully."""
    res = manager.add_belief("Socrates is a man")
    assert res["status"] == "success"
    
    # Verify in graph
    state = manager.get_current_state()
    assert any(n["properties"].get("id") == "Socrates" for n in state["nodes"])

def test_ambiguity_detection(manager):
    """
    Test that a statement triggering multiple patterns raises AmbiguityDetectedError.
    
    Sentence: 'Socrates is a teacher who causes trouble'
    Matches:
    1. 'X is a Y' (Socrates is a teacher who causes trouble)
    2. 'X causes Y' (Socrates is a teacher who causes trouble)
    """
    # Note: For this to work, our regex/logic must be broad enough.
    # 'Socrates is a teacher who causes trouble' 
    # Pattern 2 matches 'Socrates' is a 'teacher who causes trouble'
    # Pattern 8 matches 'Socrates is a teacher who' causes 'trouble'
    
    with pytest.raises(AmbiguityDetectedError) as excinfo:
        manager.add_belief("Socrates is a teacher who causes trouble")
    
    assert len(excinfo.value.possible_parses) >= 2
    types = [p["type"] for p in excinfo.value.possible_parses]
    assert "class_membership" in types
    assert "relation" in types

def test_pruning_with_axioms(manager):
    """
    Test that invalid parses are pruned according to T1 constraints.
    
    Scenario:
    - Axiom: Agents are forbidden to HARM Humans.
    - Belief: 'Socrates harms a human'
    - This is ambiguous if we also have a property 'harms a human'.
    - But since the relation 'HARM' is forbidden, that parse should be pruned?
    - Wait, pruning is for when we have MULTIPLE interpretations and want to see if some are invalid.
    """
    # 1. Seed the axiom
    manager.incorporate_axiom(DeontologicalAxiom(
        tier="T1",
        subject_type="Agent",
        relation_type="HARM",
        object_type="Human",
        constraint="FORBIDDEN",
        source_axiom="A1"
    ))
    
    # 2. Add an entity 'Alice' who is a Human
    manager.add_belief("Alice is a human")
    
    # 3. Try to add an ambiguous belief that includes a forbidden relation
    # We need a sentence that triggers a property parse AND a relation parse.
    # 'Alice is a human who harms people'
    # Parse A: Property 'human who harms people'
    # Parse B: Relation 'Alice' -[HARMS]-> 'people' (if we had a pattern for it)
    
    # Let's use a simpler one: 'The Agent harms Alice'
    # If this matches a relation AND something else.
    
    # For testing, let's manually trigger a situation where check_constraint would return BLOCKED.
    # In synthesis.py, Pattern 8 (Relational verbs) will match 'The Agent harms Alice'.
    # If we had another pattern that matched it, we'd have ambiguity.
    
    # Let's just verify that if a parse IS a forbidden relation, it's NOT in possible_parses after pruning.
    
    # Since we only have one parse for 'The Agent harms Alice', it should just proceed 
    # (but it might fail during incorporation if we check there, but add_belief only checks during pruning).
    
    # If it's the ONLY parse, it proceeds.
    res = manager.add_belief("Agent harms Alice")
    # Wait, 'Agent harms Alice' matches Pattern 8. 
    # In my refactored add_belief, if it's the only parse, it incorporates it.
    # BUT wait, I added pruning!
    # If the only parse is forbidden, valid_parses will be EMPTY.
    # And it will fallback to unstructured belief!
    
    assert res["status"] == "success"
    assert "unstructured belief" in res["message"]

def test_ambiguity_resolution_via_pruning(manager):
    """
    Test that if multiple parses exist but only one is valid, it resolves without error.
    """
    # 1. Seed axiom: Socrates cannot BE a teacher (just for testing pruning)
    # Wait, our current pruning only checks relations.
    
    # Let's use a relation:
    manager.incorporate_axiom(DeontologicalAxiom(
        tier="T1",
        subject_type="Socrates",
        relation_type="HARM",
        object_type="Student",
        constraint="FORBIDDEN",
        source_axiom="A1"
    ))
    manager.add_belief("Plato is a student")
    
    # Sentence: 'Socrates is a teacher who harms Plato'
    # Parse 1: Property 'teacher who harms Plato' (Valid)
    # Parse 2: Relation 'Socrates' -[HARM]-> 'Plato' (Invalid/Blocked)
    
    # It should resolve to Parse 1.
    res = manager.add_belief("Socrates is a teacher who harms Plato")
    assert res["status"] == "success"
    assert "Added belief" in res["message"]
    # Should NOT be unstructured
    assert "unstructured" not in res["message"]

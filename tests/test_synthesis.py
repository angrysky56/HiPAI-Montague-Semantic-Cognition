def test_synthesis(manager):
    # Clear graph first (optional but good for testing)
    manager.world_model.clear_graph()

    print("[+] Adding beliefs...")
    # Add base beliefs
    manager.add_belief("Socrates is a man")
    manager.add_belief("All men are mortal")

    # Run synthesis (this would happen in the background or triggered)
    # In our implementation, HIPAIManager.add_belief handles the logic

    # Check graph state
    state = manager.get_current_state()
    print(f"Total nodes: {len(state.get('nodes', []))}")
    print(f"Total edges: {len(state.get('edges', []))}")

    # Evaluate hypothesis
    print("[+] Evaluating hypothesis: Socrates is mortal")
    result = manager.evaluate_hypothesis("Socrates is mortal")

    print(f"Result: {result['entailment']}")

    assert result["entailment"] == "Entailed"
    print("[!] Test passed!")


def test_tense_parsing(manager):
    print(f"DEBUG: test_tense_parsing started with manager type {type(manager)}")
    manager.world_model.clear_graph()

    # Test past tense
    print("DEBUG: Adding Socrates belief")
    manager.add_belief("Socrates was a man")
    # Verify the observation node has tense="past"
    q_tense = (
        "MATCH (o) WHERE o.text_source CONTAINS 'Socrates' "
        "RETURN labels(o), o.tense"
    )
    res = manager.world_model.query_graph(q_tense)
    assert any("past" in str(row) for row in res)

    # Test future tense relation
    manager.add_belief("Alice will visit Bob")
    # Verify the edge has tense="future"
    q_visit = (
        "MATCH (a:Entity {id: 'alice'})-[r:VISIT]->(b:Entity {id: 'bob'}) "
        "RETURN r.tense"
    )
    res_edge = manager.world_model.query_graph(q_visit)
    assert res_edge[0][0] == "future"

    # Test present tense (default)
    manager.add_belief("Plato is a philosopher")
    q_plato = (
        "MATCH (o:Observation {text_source: 'Plato is a philosopher'}) "
        "RETURN o.tense"
    )
    res_pres = manager.world_model.query_graph(q_plato)
    assert res_pres[0][0] == "present"

    print("[!] Tense parsing test passed!")

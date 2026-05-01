from hipai.synthesis import HIPAIManager


def test_synthesis():
    manager = HIPAIManager(graph_name="Test_Synthesis_Graph")

    manager = HIPAIManager(graph_name="test_synthesis")

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

    for node in state.get("nodes", []):
        print(f"NODE: {node}")

    for edge in state.get("edges", []):
        print(f"EDGE: [{edge['source']}, {edge['type']}, {edge['target']}]")

    # Evaluate hypothesis
    print("[+] Evaluating hypothesis: Socrates is mortal")
    result = manager.evaluate_hypothesis("Socrates is mortal")

    print(f"Result: {result['entailment']}")

    assert result["entailment"] == "Entailed"
    print("[!] Test passed!")



def test_tense_parsing():
    manager = HIPAIManager(graph_name="test_tense")
    manager.world_model.clear_graph()

    # Test past tense
    manager.add_belief("Socrates was a man")
    # Verify the observation node has tense="past"
    res = manager.world_model.query_graph("MATCH (o) WHERE o.text_source CONTAINS 'Socrates' RETURN labels(o), o.tense")
    print(f"DEBUG: Tense parsing result: {res}")
    assert any("past" in str(row) for row in res)

    # Test future tense relation
    manager.add_belief("Alice will visit Bob")
    # Verify the edge has tense="future"
    res_edge = manager.world_model.query_graph("MATCH (a:Entity {id: 'Alice'})-[r:VISIT]->(b:Entity {id: 'Bob'}) RETURN r.tense")
    assert res_edge[0][0] == "future"

    # Test present tense (default)
    manager.add_belief("Plato is a philosopher")
    res_pres = manager.world_model.query_graph("MATCH (o:Observation {text_source: 'Plato is a philosopher'}) RETURN o.tense")
    assert res_pres[0][0] == "present"

    print("[!] Tense parsing test passed!")


if __name__ == "__main__":
    test_synthesis()
    test_tense_parsing()

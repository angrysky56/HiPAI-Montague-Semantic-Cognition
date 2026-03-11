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
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

    assert result["entailment"] == "True"
    print("[!] Test passed!")


if __name__ == "__main__":
    test_synthesis()

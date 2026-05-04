import asyncio

from hipai.synthesis import HIPAIManager


async def test_bridge():
    # HIPAIManager initializes its own WorldModel and ClaimExtractor
    hm = HIPAIManager(db_path="test_world.db")
    hm.clear_database()

    print("--- Phase 1: Aristotle is a philosopher ---")
    hm.add_belief("Aristotle is a philosopher")

    print("\n--- Phase 2: All philosophers are men ---")
    # This should create Concept_Philosopher -[:SUBCLASS_OF]-> Concept_Man
    # AND Concept_Philosopher -[:REPRESENTS]-> philosopher (Entity)
    hm.add_belief("All philosophers are men")

    print("\n--- Phase 3: All men are mortal ---")
    hm.add_belief("All men are mortal")

    print("\n--- Testing Hypothesis: Aristotle is mortal ---")
    result = hm.evaluate_hypothesis("Aristotle is mortal")
    print(f"Result: {result['entailment']}")
    print(f"Evidence: {result['evidence']}")

    # Debug: Check the graph
    print("\n--- Graph State ---")
    nodes = hm.world_model.graph.query("MATCH (n) RETURN n.id, labels(n), n.name")
    for node in nodes.result_set:
        print(node)

    edges = hm.world_model.graph.query(
        "MATCH (a)-[r]->(b) RETURN a.id, type(r), b.id, labels(a), labels(b)"
    )
    for edge in edges.result_set:
        print(edge)


if __name__ == "__main__":
    asyncio.run(test_bridge())

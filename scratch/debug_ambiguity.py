
from hipai.synthesis import HIPAIManager


def test_ambiguity_debug():
    manager = HIPAIManager()
    manager.world_model.clear_graph()
    
    print("Adding Alice Smith...")
    manager.add_belief("Alice Smith is a human")
    
    print("Searching for Alice...")
    results = manager.world_model.semantic_search("Alice", top_k=5, threshold=1.0, label="Entity")
    print(f"Results for 'Alice': {results}")
    
    # Check if Alice Smith is in the graph
    res = manager.world_model.query_graph("MATCH (n:Entity) RETURN n.id, n.name, n.embedding IS NOT NULL")
    print(f"Graph Entities: {res}")

if __name__ == "__main__":
    test_ambiguity_debug()

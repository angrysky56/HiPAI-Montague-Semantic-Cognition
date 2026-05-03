import os

from hipai.synthesis import HIPAIManager


def debug_parsing():
    manager = HIPAIManager()
    manager.world_model.clear_graph()

    # Test Existential
    manager.add_belief("Some humans are immortal")

    nodes = manager.world_model.query_graph(
        "MATCH (n) RETURN n.id, n.name, labels(n), n.prop_immortal"
    )
    edges = manager.world_model.query_graph(
        "MATCH (s)-[r]->(t) RETURN s.id, type(r), t.id"
    )

    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")


if __name__ == "__main__":
    debug_parsing()

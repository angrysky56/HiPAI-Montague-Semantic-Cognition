from hipai.synthesis import HIPAIManager
import logging

logging.basicConfig(level=logging.INFO)

def debug():
    manager = HIPAIManager()
    manager.world_model.clear_graph()
    
    print("Ingesting: Alice will visit Bob")
    manager.add_belief("Alice will visit Bob")
    
    q = "MATCH (a:Entity {id: 'alice'})-[r:VISIT]->(b:Entity {id: 'bob'}) RETURN r.tense"
    res = manager.world_model.query_graph(q)
    print(f"Result: {res}")
    
    if not res:
        print("No relation found. Let's check nodes.")
        res_nodes = manager.world_model.query_graph("MATCH (n:Entity) RETURN n.id, labels(n)")
        print(f"Nodes: {res_nodes}")
        
        res_rels = manager.world_model.query_graph("MATCH ()-[r]->() RETURN type(r), r.tense")
        print(f"Relations: {res_rels}")

if __name__ == "__main__":
    debug()

from owlready2 import *
import sys

# Mock Neo4j Driver for the Spike
class Neo4jMock:
    def __init__(self):
        self.nodes = {} 
        self.edges = set() # Use set to simulate uniqueness (idempotency)
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "-[:INFERRED_AS]->" in query:
            src = params.get("src")
            dst = params.get("dst")
            self.edges.add((src, "INFERRED_AS", dst))

def project_to_neo4j(onto, driver):
    """
    Project OWL inferences to Neo4j.
    Now includes transitive closure of types (ancestors).
    """
    print("\n[Projection] Running OWL -> Neo4j Projection...")
    
    for ind in onto.individuals():
        # Project direct and indirect class memberships
        # We walk the ancestors of each class the individual belongs to
        all_types = set()
        for cls in ind.is_a:
            if isinstance(cls, ThingClass):
                # .ancestors() returns the class and all its parents (recursive)
                for ancestor in cls.ancestors():
                    if isinstance(ancestor, ThingClass) and ancestor != Thing:
                        all_types.add(ancestor.name)
        
        # Project to Neo4j
        for type_name in all_types:
            driver.run(
                "MATCH (n:Entity {name: $src}) "
                "MERGE (c:Class {name: $dst}) "
                "MERGE (n)-[:INFERRED_AS]->(c)",
                src=ind.name, dst=type_name
            )
    
    print(f"[Projection] Projected {len(list(onto.individuals()))} entities.")

def test_projection_logic():
    print("--- Spike 004: Neo4j Projection Logic (Revised) ---")
    
    onto = get_ontology("http://test.org/projection.owl")
    with onto:
        class Entity(Thing): pass
        class Mortal(Entity): pass
        class Man(Mortal): pass # Man is a subclass of Mortal
        
        class MoralAgent(Entity): pass
        class MoralPatient(Entity): pass
        class harms(ObjectProperty):
            domain = [MoralAgent]
            range = [MoralPatient]
            
        socrates = Man("Socrates")
        pig = MoralPatient("Pig")
        socrates.harms.append(pig)

    print("Reasoning...")
    with onto:
        sync_reasoner()
        
    neo4j = Neo4jMock()
    project_to_neo4j(onto, neo4j)
    
    print("\n[1] Verifying Projection Results (Full Hierarchy):")
    socrates_types = [dst for src, type, dst in neo4j.edges if src == "Socrates"]
    print(f"  Socrates Inferred Types in Neo4j: {socrates_types}")
    
    # Expected: Man, Mortal, MoralAgent
    expected = ["Man", "Mortal", "MoralAgent"]
    all_present = all(t in socrates_types for t in expected)
    print(f"  All expected types present? {'YES ✓' if all_present else 'NO ✗'}")

if __name__ == "__main__":
    test_projection_logic()

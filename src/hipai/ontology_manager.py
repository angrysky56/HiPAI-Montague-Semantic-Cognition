import os
import logging
import owlready2
from owlready2 import get_ontology, World

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    Manages the owlready2 ontology and SQLite backend for HiPAI.
    This provides the authoritative logical layer (T1/T2).
    """

    def __init__(self, db_path: str = "world.db"):
        self.db_path = os.path.abspath(db_path)
        self.world = World(filename=self.db_path)
        self.onto = None

    def init_world(self, onto_iri: str = "http://hipai.org/ontology"):
        """
        Initializes the SQLite world model and the base ontology.
        """
        logger.info(f"Initializing world at {self.db_path}")
        
        # Load or create the ontology within our specific world
        self.onto = self.world.get_ontology(onto_iri)
        
        # Ensure the world is loaded/created
        # Note: world.load() is called automatically when using World(filename=...)
        # but we can explicitly load the ontology
        self.onto.load()
        
        return self.onto

    def seed_axioms(self):
        """
        Defines the base T1 hierarchy and core properties.
        """
        if not self.onto:
            raise ValueError("Ontology not initialized. Call init_world() first.")

        with self.onto:
            # 1. Base T1 Hierarchy
            class Entity(owlready2.Thing): pass
            class Action(Entity): pass
            class Agent(Entity): pass
            class Patient(Entity): pass

            # 2. Core Properties
            class harms(Agent >> Patient): pass
            class deceives(Agent >> Agent): pass
            class violates_agency(Agent >> Agent): pass

            # 3. Disjointness (The "Gates")
            owlready2.AllDisjoint([Action, Agent, Patient])

        logger.info("Axioms seeded successfully.")
        self.save()

    def save(self):
        """Saves the current world state to the SQLite DB."""
        self.world.save()

    def close(self):
        """Closes the world connection."""
        self.world.close()

if __name__ == "__main__":
    # Sanity check if run directly
    logging.basicConfig(level=logging.INFO)
    mgr = OntologyManager("world.db")
    onto = mgr.init_world()
    mgr.seed_axioms()
    
    # Verify seeding
    print(f"Ontology Classes: {list(onto.classes())}")
    print(f"Ontology Properties: {list(onto.properties())}")
    
    mgr.save()
    print("World saved.")

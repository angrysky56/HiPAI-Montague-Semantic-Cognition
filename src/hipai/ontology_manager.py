import os
import logging
from typing import TYPE_CHECKING, Optional

import owlready2
from owlready2 import get_ontology, World

if TYPE_CHECKING:
    from .models import Observation

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    Manages the owlready2 ontology and SQLite backend for HiPAI.
    This provides the authoritative logical layer (T1/T2).
    """

    def __init__(self, db_path: str = "world.db"):
        self.db_path = os.path.abspath(db_path)
        self.world = World(filename=self.db_path)
        self.onto = self.init_world()
        
        # Seed if classes are empty
        if not list(self.onto.classes()):
            self.seed_axioms()

    def init_world(self, onto_iri: str = "http://hipai.org/ontology"):
        """
        Initializes the SQLite world model and the base ontology.
        """
        logger.info(f"Initializing world at {self.db_path}")
        onto = self.world.get_ontology(onto_iri)
        onto.load()
        return onto

    def add_observation(self, obs: "Observation"):
        """
        Adds an observation to the OWL model.
        """
        with self.onto:
            # Pre-fetch base classes to avoid repeated attribute access
            Entity = self.onto.Entity
            
            for individual in obs.individuals:
                # 1. Create or find individual
                name = individual.id
                onto_ind = self.onto.search_one(iri=f"*{name}")
                
                # If it's a class instead of an individual, we have a name collision
                if onto_ind and not isinstance(onto_ind, owlready2.Thing):
                    # For now, just append a suffix or skip
                    name = f"{name}_ind"
                    onto_ind = self.onto.search_one(iri=f"*{name}")
                
                if onto_ind is None:
                    onto_ind = Entity(name)
                
                # 2. Add properties (classes)
                if individual.properties:
                    for prop in individual.properties:
                        prop_name = prop.replace(" ", "_")
                        # Skip if it's the same as the individual name to avoid confusion
                        if prop_name == name:
                            continue
                            
                        # Find or create class
                        cls = getattr(self.onto, prop_name, None)
                        if cls is None or not isinstance(cls, owlready2.ThingClass):
                            cls = type(prop_name, (Entity,), {})
                        
                        if cls not in onto_ind.is_a:
                            onto_ind.is_a.append(cls)

            for relation in obs.relations:
                # Handle standard relations
                if relation.target_id:
                    source_ind_model = next((i for i in obs.individuals if i.id == relation.source_id), None)
                    target_ind_model = next((i for i in obs.individuals if i.id == relation.target_id), None)
                    if source_ind_model and target_ind_model:
                        source_name = source_ind_model.name.replace(" ", "_")
                        target_name = target_ind_model.name.replace(" ", "_")
                        
                        source_onto = self.onto.search_one(iri=f"*{source_name}")
                        target_onto = self.onto.search_one(iri=f"*{target_name}")
                        
                        if source_onto and target_onto:
                            rel_name = relation.relation_type.lower()
                            rel_prop = self.onto.search_one(iri=f"*{rel_name}", type=owlready2.ObjectProperty)
                            if rel_prop is None:
                                rel_prop = type(rel_name, (owlready2.ObjectProperty,), {})
                            
                            # Add relation
                            if target_onto not in getattr(source_onto, rel_name):
                                getattr(source_onto, rel_name).append(target_onto)
                
                # Recursively add nested observations (even if we don't store the attitude relation in OWL yet)
                if relation.target_observation:
                    self.add_observation(relation.target_observation)

        self.save()

    def check_action(self, subject_name: str, relation_name: str, object_name: str) -> dict:
        """
        Checks if an action is permitted according to T1 axioms.
        Currently focused on 'harms' property.
        """
        rel_lower = relation_name.lower()
        
        # Simple reasoning check:
        # If the action is 'harms' and subject is Agent and object is Patient,
        # we check for explicit forbidden rules.
        # For Phase 2, we implement a basic structural check.
        
        is_forbidden = False
        reasoning = f"Checking if {subject_name} {rel_lower} {object_name}"
        
        if rel_lower in ["harms", "harm"]:
            # Rule: Agents should not harm Patients
            # We can check if object is an instance of Patient
            obj_ind = self.onto.search_one(iri=f"*{object_name.replace(' ', '_')}")
            if obj_ind:
                if any(isinstance(cls, type) and issubclass(cls, self.onto.Patient) for cls in obj_ind.is_a):
                    is_forbidden = True
                    reasoning += f" | Object {object_name} is a Patient. HARMS is forbidden."

        return {
            "permitted": not is_forbidden,
            "blocking_axiom": "T1-HARMS-PROTECTION" if is_forbidden else None,
            "tier": "T1",
            "reasoning": reasoning
        }

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
            class harm(Agent >> Patient): pass
            class deceive(Agent >> Agent): pass
            class violate_agency(Agent >> Agent): pass

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

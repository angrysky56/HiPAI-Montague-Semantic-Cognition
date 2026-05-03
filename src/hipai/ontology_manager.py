"""
Ontology management for HiPAI using owlready2.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import owlready2
from owlready2 import World

if TYPE_CHECKING:
    from .models import Observation

logger = logging.getLogger(__name__)


class OntologyManager:
    """
    Manages the owlready2 ontology and SQLite backend for HiPAI.
    This provides the authoritative logical layer (T1/T2).
    """

    def __init__(self, db_path: str = "world.db"):
        self.db_path = (
            db_path if db_path == ":memory:" else str(Path(db_path).resolve())
        )
        self.world = World(filename=self.db_path)
        self.onto = self.init_world()

        # Seed if classes are empty
        if not list(self.onto.classes()):
            self.seed_axioms()

    def init_world(self, onto_iri: str = "http://hipai.org/ontology"):
        """
        Initializes the SQLite world model and the base ontology.
        """
        logger.info("Initializing world at %s", self.db_path)
        ontology = self.world.get_ontology(onto_iri)
        ontology.load()
        return ontology

    def add_observation(self, obs: "Observation") -> owlready2.Thing | None:
        """
        Adds an observation to the OWL model and returns the created instance.
        """
        with self.onto:
            # 1. Create individuals
            for individual in obs.individuals:
                name = individual.id.replace(" ", "_")
                onto_ind = self.onto.search_one(iri=f"*{name}")
                if onto_ind is None:
                    onto_ind = self.onto.Entity(name)

                if individual.properties:
                    for prop in individual.properties:
                        prop_name = prop.replace(" ", "_")
                        # Case-insensitive lookup
                        cls = None
                        for c in self.onto.classes():
                            if c.name.lower() == prop_name.lower():
                                cls = c
                                break

                        if cls is None:
                            # Create new class if not found
                            cls = type(f"Concept_{prop_name}", (self.onto.Entity,), {})
                        if cls not in onto_ind.is_a:
                            onto_ind.is_a.append(cls)

            # 2. Create the Observation instance for THIS level
            main_obs_ind = self.onto.Observation()

            # 3. Handle relations
            for relation in obs.relations:
                # Standard relation
                if relation.target_id:
                    source_ind = self.onto.search_one(
                        iri=f"*{relation.source_id.replace(' ', '_')}"
                    )
                    target_ind = self.onto.search_one(
                        iri=f"*{relation.target_id.replace(' ', '_')}"
                    )

                    if source_ind and target_ind:
                        rel_name = relation.relation_type.lower()

                        if rel_name == "is_a":
                            # Handle class membership: source is an instance of target
                            # Ensure we have a class for the target
                            target_class = None
                            target_name = relation.target_id.replace(" ", "_")
                            for c in self.onto.classes():
                                if c.name.lower() == target_name.lower():
                                    target_class = c
                                    break
                            if target_class is None:
                                # Create class if not found
                                target_class = type(
                                    f"Concept_{target_name}", (self.onto.Entity,), {}
                                )

                            if target_class not in source_ind.is_a:
                                source_ind.is_a.append(target_class)
                        else:
                            # Support both lowercase and CapWords property names
                            rel_prop = getattr(self.onto, rel_name, None)

                            if rel_prop is None:
                                rel_prop = type(
                                    rel_name, (owlready2.ObjectProperty,), {}
                                )

                            # Use rel_prop.python_name if available, else use rel_prop.name
                            prop_attr = getattr(rel_prop, "python_name", rel_prop.name)
                            if target_ind not in getattr(source_ind, prop_attr):
                                getattr(source_ind, prop_attr).append(target_ind)

                        # Link this observation to its components if it's the root fact
                        if main_obs_ind.source is None:
                            main_obs_ind.source = source_ind
                        if main_obs_ind.target is None:
                            main_obs_ind.target = target_ind
                        main_obs_ind.relation_type = rel_name

                # Nested (recursive) relation
                if relation.target_observation:
                    inner_obs_ind = self.add_observation(relation.target_observation)
                    source_ind = self.onto.search_one(
                        iri=f"*{relation.source_id.replace(' ', '_')}"
                    )

                    if source_ind and inner_obs_ind:
                        main_obs_ind.source = source_ind
                        main_obs_ind.nested_observation = inner_obs_ind
                        main_obs_ind.relation_type = relation.relation_type.lower()

        self.save()
        return main_obs_ind

    def check_action(
        self, subject_name: str, relation_name: str, object_name: str
    ) -> dict:
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
            if obj_ind and any(
                isinstance(cls, owlready2.ThingClass)
                and issubclass(cls, self.onto.Patient)
                for cls in obj_ind.is_a
            ):
                is_forbidden = True
                reasoning += (
                    f" | Object {object_name} is a Patient. HARMS is forbidden."
                )

        return {
            "permitted": not is_forbidden,
            "blocking_axiom": "T1-HARMS-PROTECTION" if is_forbidden else None,
            "tier": "T1",
            "reasoning": reasoning,
        }

    def seed_axioms(self):
        """
        Defines the base T1 hierarchy and core properties.
        """
        if not self.onto:
            raise ValueError("Ontology not initialized. Call init_world() first.")

        with self.onto:
            # 1. Base T1 Hierarchy
            class Entity(owlready2.Thing):
                """Base class for all entities in the world model."""

            class Action(Entity):
                """Represents an action performed by an agent."""

            class Agent(Entity):
                """An entity capable of performing actions."""

            class Patient(Entity):
                """An entity that can be the recipient of an action."""

            class Observation(Entity):
                """Represents a cognitive observation or belief."""

            # 2. Core Properties
            class Harm(Agent >> Patient):
                """Property representing an agent harming a patient."""

                python_name = "harm"

            class Deceive(Agent >> Agent):
                """Property representing an agent deceiving another agent."""

                python_name = "deceive"

            class ViolateAgency(Agent >> Agent):
                """Property representing an agent violating another's agency."""

                python_name = "violate_agency"

            # 3. Recursive Cognitive Properties
            class Source(Observation >> Agent, owlready2.FunctionalProperty):
                """The agent who is the source of an observation."""

                python_name = "source"

            class Target(Observation >> Entity, owlready2.FunctionalProperty):
                """The entity that is the target of an observation."""

                python_name = "target"

            class NestedObservation(
                Observation >> Observation, owlready2.FunctionalProperty
            ):
                """A recursive link to another observation."""

                python_name = "nested_observation"

            class RelationType(Observation >> str, owlready2.FunctionalProperty):
                """The type of relation described in the observation."""

                python_name = "relation_type"

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

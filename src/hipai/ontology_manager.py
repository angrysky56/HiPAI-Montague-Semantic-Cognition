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
        import time

        self.db_path = (
            db_path if db_path == ":memory:" else str(Path(db_path).resolve())
        )

        # Retry logic for locked database
        retries = 5
        last_err = None
        while retries > 0:
            try:
                self.world = World(filename=self.db_path)
                # Set a longer busy timeout (5 seconds) for future operations
                self.world.graph.db.execute("PRAGMA busy_timeout = 5000")
                break
            except Exception as e:
                last_err = e
                if "locked" in str(e).lower() and retries > 1:
                    logger.warning("Database %s is locked, retrying...", self.db_path)
                    time.sleep(1)
                    retries -= 1
                else:
                    raise last_err

        self.onto = self.init_world()

        # Seed if classes are empty
        if not list(self.onto.classes()):
            self.seed_axioms()

    def close(self):
        """
        Closes the SQLite world backend.
        """
        if hasattr(self, "world"):
            try:
                self.world.close()
            except Exception as e:
                logger.warning("Error closing world: %s", e)

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
            # 1. Create individuals or class subsumptions
            for individual in obs.individuals:
                name = individual.id.replace(" ", "_")

                if individual.quantifier == "all":
                    # This represents a universal rule: All X are Y
                    # Find or create class for X
                    base_cls = None
                    for c in self.onto.classes():
                        if (
                            c.name.lower() == name.lower()
                            or c.name.lower() == f"concept_{name.lower()}"
                        ):
                            base_cls = c
                            break
                    if base_cls is None:
                        base_cls = type(
                            f"Concept_{name.capitalize()}", (self.onto.Entity,), {}
                        )

                    if individual.properties:
                        for prop in individual.properties:
                            prop_name = prop.replace(" ", "_")
                            target_cls = None
                            for c in self.onto.classes():
                                if (
                                    c.name.lower() == prop_name.lower()
                                    or c.name.lower() == f"concept_{prop_name.lower()}"
                                ):
                                    target_cls = c
                                    break
                            if target_cls is None:
                                target_cls = type(
                                    f"Concept_{prop_name.capitalize()}",
                                    (self.onto.Entity,),
                                    {},
                                )

                            if target_cls not in base_cls.is_a:
                                base_cls.is_a.append(target_cls)
                    continue

                onto_ind = self.onto.search_one(iri=f"*{name}")
                if onto_ind is None:
                    onto_ind = self.onto.Entity(name)

                if individual.properties:
                    for prop in individual.properties:
                        prop_name = prop.replace(" ", "_")
                        # Case-insensitive lookup
                        cls = None
                        for c in self.onto.classes():
                            if (
                                c.name.lower() == prop_name.lower()
                                or c.name.lower() == f"concept_{prop_name.lower()}"
                            ):
                                cls = c
                                break

                        if cls is None:
                            # Create new class if not found
                            cls = type(
                                f"Concept_{prop_name.capitalize()}",
                                (self.onto.Entity,),
                                {},
                            )
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
                    rel_name = relation.relation_type.lower()

                    if rel_name == "is_a":
                        # Handle class membership: source is an instance of target class
                        target_class = None
                        target_name = relation.target_id.replace(" ", "_")
                        # Try to find class
                        target_class = getattr(
                            self.onto, target_name.capitalize(), None
                        )
                        if not target_class:
                            for c in self.onto.classes():
                                if (
                                    c.name.lower() == target_name.lower()
                                    or c.name.lower()
                                    == f"concept_{target_name.lower()}"
                                ):
                                    target_class = c
                                    break
                        if target_class is None:
                            # Create class if not found
                            target_class = type(
                                f"Concept_{target_name.capitalize()}",
                                (self.onto.Entity,),
                                {},
                            )

                        if source_ind and target_class not in source_ind.is_a:
                            source_ind.is_a.append(target_class)

                    else:
                        target_ind = self.onto.search_one(
                            iri=f"*{relation.target_id.replace(' ', '_')}"
                        )
                        if source_ind and target_ind:
                            # Support both lowercase and CapWords property names
                            rel_prop = getattr(self.onto, rel_name, None)
                            if rel_prop is None:
                                rel_prop = type(
                                    rel_name, (owlready2.ObjectProperty,), {}
                                )
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
        self,
        subject_name: str,
        relation_name: str,
        object_name: str,
        constraints: list[dict] | None = None,
    ) -> dict:
        """
        Checks if an action is permitted according to T1 axioms.
        Uses Owlready2 to verify if the subject and object match the
        categories defined in the constraints.
        """
        rel_lower = relation_name.lower()
        is_forbidden = False
        blocking_axiom = None
        reasoning = f"Checking action: {subject_name} {rel_lower} {object_name}"

        # If no constraints provided, use the hardcoded baseline for backward compatibility
        # but the goal is to always pass constraints from WorldModel.
        if not constraints:
            # Baseline: Agents should not harm Patients
            if rel_lower in ["harms", "harm"]:
                obj_ind = self.onto.search_one(iri=f"*{object_name.replace(' ', '_')}")
                if obj_ind:
                    # Force a list to avoid iterator issues
                    classes = list(obj_ind.is_a)
                    if any(
                        isinstance(cls, owlready2.ThingClass)
                        and issubclass(cls, self.onto.Patient)
                        for cls in classes
                    ):
                        is_forbidden = True
                        blocking_axiom = "T1-HARMS-PROTECTION"
                        reasoning += (
                            f" | [Baseline] Object {object_name} is a Patient. "
                            "HARMS is forbidden."
                        )
        else:
            for ax in constraints:
                # 1. Match relation type (normalized)
                ax_rel = ax.get("relation_type", "").lower()
                if ax_rel != rel_lower and ax_rel != rel_lower + "s":
                    # Simple heuristic: 'harm' matches 'harms'
                    if not (
                        (ax_rel == "harm" and rel_lower == "harms")
                        or (ax_rel == "harms" and rel_lower == "harm")
                    ):
                        continue

                # 2. Check if object matches object_type
                # We prioritize object_type for T1 protections (Patient-centric)
                obj_type = ax.get("object_type")
                if obj_type:
                    obj_ind = self.onto.search_one(
                        iri=f"*{object_name.replace(' ', '_')}"
                    )
                    # Get the class from the ontology
                    protected_cls = getattr(self.onto, obj_type, None)
                    if not protected_cls:
                        # Try case-insensitive search and also Concept_ prefix
                        for c in self.onto.classes():
                            if (
                                c.name.lower() == obj_type.lower()
                                or c.name.lower() == f"concept_{obj_type.lower()}"
                            ):
                                protected_cls = c
                                break

                    if obj_ind and protected_cls:

                        # Recursive check for class membership
                        is_match = isinstance(obj_ind, protected_cls)
                        if not is_match:
                            # Owlready2 sometimes needs manual check of ancestors for dynamic classes
                            for cls in obj_ind.is_a:
                                if protected_cls == cls or (
                                    isinstance(cls, owlready2.ThingClass)
                                    and protected_cls in cls.ancestors()
                                ):
                                    is_match = True
                                    break

                        if is_match and ax.get("constraint") == "FORBIDDEN":
                            is_forbidden = True
                            blocking_axiom = ax.get("source_axiom")
                            reasoning += (
                                f" | Violation of {blocking_axiom}: "
                                f"{object_name} is a {obj_type}."
                            )
                            break

        return {
            "permitted": not is_forbidden,
            "blocking_axiom": blocking_axiom,
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

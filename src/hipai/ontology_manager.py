"""
Ontology management for HiPAI using owlready2.
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import owlready2
from owlready2 import World

from ._utils import canonical_concept_name, lemmatize_verb

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

        # Retry logic for locked database
        retries = 5
        while retries > 0:
            try:
                self.world = World(filename=self.db_path)
                # Set a longer busy timeout (5 seconds) for future operations
                self.world.graph.db.execute("PRAGMA busy_timeout = 5000")
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if "locked" in str(e).lower() and retries > 1:
                    logger.warning("Database %s is locked, retrying...", self.db_path)
                    time.sleep(1)
                    retries -= 1
                else:
                    raise e
            except Exception as e:
                logger.exception("Unexpected error initializing world: %s", e)
                raise e

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
            except (sqlite3.Error, RuntimeError) as e:
                logger.error("Error closing world: %s", e, exc_info=True)

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
                # Use lemmatised .name (not raw .id) for the OWL entity name so
                # that OWL class hierarchies stay canonical (e.g. "man" not "men").
                name = individual.name.replace(" ", "_")

                if individual.quantifier == "all":
                    # This represents a universal rule: All X are Y.
                    # Use lemmatised .name so OWL class names stay canonical.
                    owl_name = individual.name.replace(" ", "_")
                    base_cls = None
                    for c in self.onto.classes():
                        if (
                            c.name.lower() == owl_name.lower()
                            or c.name.lower() == f"concept_{owl_name.lower()}"
                        ):
                            base_cls = c
                            break
                    if base_cls is None:
                        base_cls = type(
                            canonical_concept_name(individual.name),
                            (self.onto.Entity,),
                            {},
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
                                    canonical_concept_name(prop),
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
                            # Create new class if not found - use canonical helper
                            cls = type(
                                canonical_concept_name(prop),
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
                    # Resolve source ID to its lemmatised name from the observation
                    source_name = relation.source_id.replace(" ", "_")
                    source_obj = next(
                        (i for i in obs.individuals if i.id == relation.source_id),
                        None,
                    )
                    if source_obj:
                        source_name = source_obj.name.replace(" ", "_")

                    source_ind = self.onto.search_one(iri=f"*{source_name}")
                    rel_name = relation.relation_type.lower()

                    if rel_name == "is_a":
                        # Handle class membership: source is an instance of target class
                        target_name = relation.target_id.replace(" ", "_")
                        target_obj = next(
                            (i for i in obs.individuals if i.id == relation.target_id),
                            None,
                        )
                        if target_obj:
                            target_name = target_obj.name.replace(" ", "_")

                        # Try to find class using canonical Concept_ name first
                        target_class = getattr(
                            self.onto, canonical_concept_name(target_name), None
                        )
                        if not isinstance(target_class, owlready2.ThingClass):
                            target_class = None

                        if not target_class:
                            # Robust matching: lowercase and strip underscores
                            norm_target = target_name.lower().replace("_", "")
                            for c in self.onto.classes():
                                norm_c = (
                                    c.name.lower()
                                    .replace("_", "")
                                    .replace("concept", "")
                                )
                                if norm_c == norm_target:
                                    target_class = c
                                    break
                        if target_class is None:
                            # Create class if not found — use canonical helper
                            # If the name matches a seeded base class (e.g. Patient),
                            # inherit from it to maintain baseline protections.
                            base_parent = getattr(
                                self.onto, target_name.capitalize(), self.onto.Entity
                            )
                            if not isinstance(base_parent, owlready2.ThingClass):
                                base_parent = self.onto.Entity

                            target_class = type(
                                canonical_concept_name(target_name),
                                (base_parent,),
                                {},
                            )

                        if (
                            source_ind
                            and target_class
                            and isinstance(target_class, owlready2.ThingClass)
                            and target_class not in source_ind.is_a
                        ):
                            source_ind.is_a.append(target_class)

                    else:
                        target_name = relation.target_id.replace(" ", "_")
                        target_obj = next(
                            (i for i in obs.individuals if i.id == relation.target_id),
                            None,
                        )
                        if target_obj:
                            target_name = target_obj.name.replace(" ", "_")

                        target_ind = self.onto.search_one(iri=f"*{target_name}")
                        if (
                            source_ind
                            and target_ind
                            and isinstance(source_ind, owlready2.Thing)
                            and isinstance(target_ind, owlready2.Thing)
                        ):
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

                    source_name = relation.source_id.replace(" ", "_")
                    source_obj = next(
                        (i for i in obs.individuals if i.id == relation.source_id),
                        None,
                    )
                    if source_obj:
                        source_name = source_obj.name.replace(" ", "_")

                    source_ind = self.onto.search_one(iri=f"*{source_name}")

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
        rel_lemma = lemmatize_verb(relation_name)
        is_forbidden = False
        blocking_axiom = None
        reasoning = f"Checking action: {subject_name} {rel_lemma} {object_name}"

        if not constraints:
            return {"permitted": True, "reasoning": "No constraints to check."}

        for ax in constraints:
            # 1. Match relation type (lemmatised)
            ax_rel = lemmatize_verb(ax.get("relation_type", ""))
            if ax_rel != rel_lemma:
                continue

            # 2. Check if object matches object_type
            # We prioritize object_type for T1 protections (Patient-centric)
            obj_type_raw = ax.get("object_type")
            if obj_type_raw:
                # Lookup with both raw and canonical names
                protected_cls = getattr(self.onto, obj_type_raw, None)
                if not protected_cls:
                    protected_cls = getattr(
                        self.onto, canonical_concept_name(obj_type_raw), None
                    )

                if not protected_cls:
                    # Robust matching: lowercase and strip underscores
                    norm_obj = (
                        obj_type_raw.lower().replace("_", "").replace("concept", "")
                    )
                    for c in self.onto.classes():
                        norm_c = c.name.lower().replace("_", "").replace("concept", "")
                        if norm_c == norm_obj:
                            protected_cls = c
                            break

                obj_ind = self.onto.search_one(iri=f"*{object_name}")
                if not obj_ind:
                    # Try case-insensitive
                    for ind in self.onto.individuals():
                        if ind.name.lower() == object_name.lower():
                            obj_ind = ind
                            break

                if obj_ind and protected_cls:
                    # Recursive check for class membership
                    try:
                        is_match = isinstance(obj_ind, protected_cls)
                    except TypeError:
                        is_match = False

                    if not is_match:
                        for cls in obj_ind.is_a:
                            if protected_cls == cls or (
                                isinstance(cls, owlready2.ThingClass)
                                and protected_cls in cls.ancestors()
                            ):
                                is_match = True
                                break

                    # 3. Check if subject matches subject_type
                    subj_type_raw = ax.get("subject_type")
                    if is_match and subj_type_raw and subj_type_raw != "Any":
                        subj_cls = getattr(self.onto, subj_type_raw, None)
                        if not subj_cls:
                            subj_cls = getattr(
                                self.onto,
                                canonical_concept_name(subj_type_raw),
                                None,
                            )

                        if not subj_cls:
                            norm_subj = (
                                subj_type_raw.lower()
                                .replace("_", "")
                                .replace("concept", "")
                            )
                            for c in self.onto.classes():
                                norm_c = (
                                    c.name.lower()
                                    .replace("_", "")
                                    .replace("concept", "")
                                )
                                if norm_c == norm_subj:
                                    subj_cls = c
                                    break

                        subj_ind = self.onto.search_one(iri=f"*{subject_name}")
                        if not subj_ind:
                            for ind in self.onto.individuals():
                                if ind.name.lower() == subject_name.lower():
                                    subj_ind = ind
                                    break

                        subj_match = False
                        if subj_ind and subj_cls:
                            try:
                                if isinstance(subj_ind, subj_cls):
                                    subj_match = True
                                else:
                                    for cls in subj_ind.is_a:
                                        if subj_cls == cls or (
                                            isinstance(cls, owlready2.ThingClass)
                                            and subj_cls in cls.ancestors()
                                        ):
                                            subj_match = True
                                            break
                            except TypeError:
                                subj_match = False
                        elif subj_cls and (
                            subject_name.lower() == subj_type_raw.lower()
                            or subject_name.lower()
                            == canonical_concept_name(subj_type_raw).lower()
                        ):
                            subj_match = True

                        if not subj_match:
                            is_match = False

                    if is_match and ax.get("constraint") == "FORBIDDEN":
                        is_forbidden = True
                        blocking_axiom = ax.get("source_axiom")
                        reasoning += (
                            f" | Violation of {blocking_axiom}: "
                            f"{subject_name} is a {subj_type_raw} and "
                            f"{object_name} is a {obj_type_raw}."
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
        # Ensure classes are referenced to satisfy linters
        _ = [
            Entity,
            Action,
            Agent,
            Patient,
            Observation,
            Harm,
            Deceive,
            ViolateAgency,
            Source,
            Target,
            NestedObservation,
            RelationType,
        ]
        self.save()

    def save(self):
        """Saves the current world state to the SQLite DB."""
        self.world.save()


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

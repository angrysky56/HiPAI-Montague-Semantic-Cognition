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

from ._utils import canonical_concept_name

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
        last_err = None
        while retries > 0:
            try:
                self.world = World(filename=self.db_path)
                # Set a longer busy timeout (5 seconds) for future operations
                self.world.graph.db.execute("PRAGMA busy_timeout = 5000")
                break
            except (sqlite3.Error, Exception) as e:
                last_err = e
                if "locked" in str(e).lower() and retries > 1:
                    logger.warning("Database %s is locked, retrying...", self.db_path)
                    time.sleep(1)
                    retries -= 1
                else:
                    raise e from None

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
            except (sqlite3.Error, Exception) as e:
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

                        # Try to find class
                        target_class = getattr(
                            self.onto, target_name.capitalize(), None
                        )
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
                            # Create class if not found — use canonical helper to
                            # prevent double-prefix and enforce TitleCase
                            target_class = type(
                                canonical_concept_name(target_name),
                                (self.onto.Entity,),
                                {},
                            )

                        if source_ind and target_class not in source_ind.is_a:
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
        rel_lower = relation_name.lower()
        is_forbidden = False
        blocking_axiom = None
        reasoning = f"Checking action: {subject_name} {rel_lower} {object_name}"

        # If no constraints provided, use the hardcoded baseline for backward
        # compatibility but the goal is to always pass constraints from WorldModel.
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
                if (
                    ax_rel != rel_lower
                    and ax_rel != rel_lower + "s"
                    and not (
                        (ax_rel == "harm" and rel_lower == "harms")
                        or (ax_rel == "harms" and rel_lower == "harm")
                    )
                ):
                    continue

                # 2. Check if object matches object_type
                # We prioritize object_type for T1 protections (Patient-centric)
                obj_type = ax.get("object_type")
                if obj_type:
                    obj_ind = self.onto.search_one(iri=f"*{object_name}")
                    if not obj_ind:
                        # Try case-insensitive
                        for ind in self.onto.individuals():
                            if ind.name.lower() == object_name.lower():
                                obj_ind = ind
                                break

                    # Get the class from the ontology
                    protected_cls = getattr(self.onto, obj_type, None)
                    if not protected_cls:
                        # Robust matching: lowercase and strip underscores
                        norm_obj = obj_type.lower().replace("_", "")
                        for c in self.onto.classes():
                            norm_c = (
                                c.name.lower().replace("_", "").replace("concept", "")
                            )
                            if norm_c == norm_obj:
                                protected_cls = c
                                break

                    if obj_ind and protected_cls:

                        # Recursive check for class membership.
                        # Guard isinstance() — protected_cls must be a Python type.
                        # If owlready2 returned an individual or property instead of a
                        # class, isinstance() raises TypeError; treat as no-match.
                        try:
                            is_match = isinstance(obj_ind, protected_cls)
                        except TypeError:
                            logger.debug(
                                "isinstance check skipped: protected_cls %r is not a "
                                "Python type (owlready2 returned non-class object).",
                                protected_cls,
                            )
                            is_match = False
                        if not is_match:
                            # Owlready2 sometimes needs manual check of ancestors
                            # for dynamic classes
                            for cls in obj_ind.is_a:
                                if protected_cls == cls or (
                                    isinstance(cls, owlready2.ThingClass)
                                    and protected_cls in cls.ancestors()
                                ):
                                    is_match = True
                                    break

                        # 3. Check if subject matches subject_type
                        subj_type = ax.get("subject_type")
                        if is_match and subj_type and subj_type != "Any":
                            subj_ind = self.onto.search_one(iri=f"*{subject_name}")
                            if not subj_ind:
                                # Try case-insensitive
                                for ind in self.onto.individuals():
                                    if ind.name.lower() == subject_name.lower():
                                        subj_ind = ind
                                        break

                            # Find the subject class
                            subj_cls = getattr(self.onto, subj_type, None)
                            if not subj_cls:
                                norm_subj = subj_type.lower().replace("_", "")
                                for c in self.onto.classes():
                                    norm_c = (
                                        c.name.lower()
                                        .replace("_", "")
                                        .replace("concept", "")
                                    )
                                    if norm_c == norm_subj:
                                        subj_cls = c
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
                            elif subj_cls and subject_name.lower() == subj_type.lower():
                                # Handle case where individual isn't in OWL yet but name matches
                                # This is common for "Agent" or seeded concepts.
                                subj_match = True

                            if not subj_match:
                                is_match = False

                        if is_match and ax.get("constraint") == "FORBIDDEN":
                            is_forbidden = True
                            blocking_axiom = ax.get("source_axiom")
                            reasoning += (
                                f" | Violation of {blocking_axiom}: "
                                f"{subject_name} is a {subj_type} and "
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

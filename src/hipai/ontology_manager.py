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

        # Pre-configure WAL mode at the file level BEFORE owlready2 opens the
        # database. WAL mode is a persistent file-level setting, so setting it
        # via a separate connection that is immediately closed ensures owlready2
        # opens in WAL mode (rather than rollback-journal mode, which takes
        # EXCLUSIVE locks and blocks all other connections).
        if self.db_path != ":memory:":
            try:
                pre_conn = sqlite3.connect(self.db_path, timeout=10.0)
                pre_conn.execute("PRAGMA busy_timeout = 10000")
                mode = pre_conn.execute("PRAGMA journal_mode = WAL").fetchone()
                pre_conn.commit()
                pre_conn.close()
                if mode and mode[0] != "wal":
                    logger.warning(
                        "WAL mode not activated for %s (current: %s). "
                        "Another process may hold an exclusive lock.",
                        self.db_path,
                        mode[0] if mode else "unknown",
                    )
            except sqlite3.Error as e:
                logger.warning("Pre-WAL setup failed for %s: %s", self.db_path, e)

        # Retry logic for locked database
        retries = 5
        attempt = 0
        last_err = None
        while attempt < retries:
            try:
                self.world = World(filename=self.db_path)
                self.world.graph.db.execute("PRAGMA busy_timeout = 10000")
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                last_err = e
                attempt += 1
                if "locked" in str(e).lower() and attempt < retries:
                    sleep_time = min(attempt * 0.5, 3)
                    logger.warning(
                        "Database %s is locked (attempt %d/%d), retrying in %.1fs...",
                        self.db_path,
                        attempt,
                        retries,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Database %s failed to open after %d retries. "
                        "Check for orphan processes: lsof %s",
                        self.db_path,
                        attempt,
                        self.db_path,
                    )
                    raise last_err from e
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

    def get_onto_class(self, name: str) -> owlready2.ThingClass | None:
        """
        Retrieves a class from the ontology by name, trying both raw and canonical names.
        """
        if not name:
            return None

        # 1. Try canonical name
        canon_name = canonical_concept_name(name)
        cls = getattr(self.onto, canon_name, None)
        if isinstance(cls, owlready2.ThingClass):
            return cls

        # 2. Try raw name
        cls = getattr(self.onto, name, None)
        if isinstance(cls, owlready2.ThingClass):
            return cls

        # 3. Robust matching (fallback) - case-insensitive
        norm_name = name.lower().replace("_", "").replace("concept", "")
        for c in self.onto.classes():
            norm_c = c.name.lower().replace("_", "").replace("concept", "")
            if norm_c == norm_name:
                return c

        return None

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
                    # 1. Try to find existing class
                    canon_base = canonical_concept_name(individual.name)
                    base_cls = getattr(self.onto, canon_base, None)
                    if not isinstance(base_cls, owlready2.ThingClass):
                        # Robust matching fallback
                        for c in self.onto.classes():
                            if c.name.lower() == canon_base.lower():
                                base_cls = c
                                break

                    if base_cls is None:
                        base_cls = type(
                            canon_base,
                            (self.onto.Concept_Entity,),
                            {},
                        )

                    if individual.properties:
                        for prop in individual.properties:
                            canon_target = canonical_concept_name(prop)
                            target_cls = getattr(self.onto, canon_target, None)
                            if not isinstance(target_cls, owlready2.ThingClass):
                                for c in self.onto.classes():
                                    if c.name.lower() == canon_target.lower():
                                        target_cls = c
                                        break
                            if target_cls is None:
                                target_cls = type(
                                    canon_target,
                                    (self.onto.Concept_Entity,),
                                    {},
                                )

                            # Prevent inheritance cycles (self-inheritance or existing ancestor)
                            if (
                                target_cls != base_cls
                                and target_cls not in base_cls.is_a
                                and target_cls not in base_cls.ancestors()
                            ):
                                base_cls.is_a.append(target_cls)
                    continue

                onto_ind = self.onto.search_one(iri=f"*{name}")
                if onto_ind is None:
                    onto_ind = self.onto.Concept_Entity(name)

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
                            with self.onto:
                                cls = type(
                                    canonical_concept_name(prop),
                                    (self.onto.Concept_Entity,),
                                    {},
                                )
                        if cls not in onto_ind.is_a:
                            onto_ind.is_a.append(cls)

            # 2. Create the Observation instance for THIS level
            main_obs_ind = self.onto.Concept_Observation()

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

                            if not target_class:
                                base_parent = self.onto.Concept_Entity
                                with self.onto:
                                    target_class = type(
                                        canonical_concept_name(target_name),
                                        (base_parent,),
                                        {},
                                    )

                        # 2. Check source
                        source_obj = next(
                            (i for i in obs.individuals if i.id == relation.source_id),
                            None,
                        )

                        if source_obj and source_obj.quantifier == "all":
                            # Universal Relation: All X are Y -> Concept_X is a subclass of Concept_Y
                            source_class = self.get_onto_class(source_obj.name)
                            if not source_class:
                                with self.onto:
                                    source_class = type(
                                        canonical_concept_name(source_obj.name),
                                        (self.onto.Concept_Entity,),
                                        {},
                                    )

                            if (
                                source_class
                                and target_class
                                and source_class != target_class
                                and target_class not in source_class.is_a
                                and target_class not in source_class.ancestors()
                            ):
                                source_class.is_a.append(target_class)
                                logger.info(
                                    "Universal Rule: %s IS_A %s",
                                    source_class.name,
                                    target_class.name,
                                )
                        else:
                            # Individual Relation: Ty is a Human -> ty is an instance of Concept_Human
                            source_ind = self.onto.search_one(
                                iri=f"*{relation.source_id}"
                            )
                            if not source_ind and source_obj:
                                source_name = source_obj.name
                                source_ind = self.onto.search_one(iri=f"*{source_name}")

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
            obj_type_raw = ax.get("object_type")
            if obj_type_raw:
                protected_cls = self.get_onto_class(obj_type_raw)
                obj_ind = self.onto.search_one(iri=f"*{object_name}")
                if not obj_ind:
                    for ind in self.onto.individuals():
                        if ind.name.lower() == object_name.lower():
                            obj_ind = ind
                            break

                if obj_ind and protected_cls:
                    try:
                        is_match = isinstance(obj_ind, protected_cls)
                        if not is_match:
                            for cls in obj_ind.is_a:
                                if protected_cls == cls or (
                                    isinstance(cls, owlready2.ThingClass)
                                    and protected_cls in cls.ancestors()
                                ):
                                    is_match = True
                                    break
                    except TypeError as e:
                        logger.error("TypeError in check_action: %s", e)
                        is_match = False

                    # 3. Check if subject matches subject_type
                    subj_type_raw = ax.get("subject_type")
                    if is_match and subj_type_raw and subj_type_raw != "Any":
                        subj_cls = self.get_onto_class(subj_type_raw)
                        subj_ind = self.onto.search_one(iri=f"*{subject_name}")
                        if not subj_ind:
                            for ind in self.onto.individuals():
                                if ind.name.lower() == subject_name.lower():
                                    subj_ind = ind
                                    break

                        # Check if subject name matches the type name (Canonical)
                        name_match = (
                            subject_name.lower() == subj_type_raw.lower()
                            or canonical_concept_name(subject_name).lower()
                            == subj_type_raw.lower()
                        )

                        subj_match = False
                        if name_match:
                            subj_match = True
                        elif subj_ind and subj_cls:
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
            # 1. Base T1 Hierarchy (Canonicalized)
            class Concept_Entity(owlready2.Thing):
                """Base class for all entities in the world model."""

            class Concept_Action(Concept_Entity):
                """Represents an action performed by an agent."""

            class Concept_Agent(Concept_Entity):
                """An entity capable of performing actions."""

            class Concept_Patient(Concept_Entity):
                """An entity that can be the recipient of an action."""

            class Concept_Observation(Concept_Entity):
                """Represents a cognitive observation or belief."""

            # 2. Core Properties
            class Harm(Concept_Agent >> Concept_Patient):
                """Property representing an agent harming a patient."""

                python_name = "harm"

            class Deceive(Concept_Agent >> Concept_Agent):
                """Property representing an agent deceiving another agent."""

                python_name = "deceive"

            class ViolateAgency(Concept_Agent >> Concept_Agent):
                """Property representing an agent violating another's agency."""

                python_name = "violate_agency"

            # 3. Recursive Cognitive Properties
            class Source(
                Concept_Observation >> Concept_Agent, owlready2.FunctionalProperty
            ):
                """The agent who is the source of an observation."""

                python_name = "source"

            class Target(
                Concept_Observation >> Concept_Entity, owlready2.FunctionalProperty
            ):
                """The entity that is the target of an observation."""

                python_name = "target"

            class NestedObservation(
                Concept_Observation >> Concept_Observation, owlready2.FunctionalProperty
            ):
                """A recursive link to another observation."""

                python_name = "nested_observation"

            class RelationType(
                Concept_Observation >> str, owlready2.FunctionalProperty
            ):
                """The type of relation described in the observation."""

                python_name = "relation_type"

            # 3. Disjointness (The "Gates")
            owlready2.AllDisjoint([Concept_Action, Concept_Agent, Concept_Patient])

        logger.info("Axioms seeded successfully.")
        # Ensure classes are referenced to satisfy linters
        _ = [
            Concept_Entity,
            Concept_Action,
            Concept_Agent,
            Concept_Patient,
            Concept_Observation,
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

    def clear_ontology(self):
        """
        Truly resets the ontology by clearing the world and re-seeding.
        """
        logger.info("Clearing ontology at %s", self.db_path)
        # Close current world
        self.world.close()

        # Delete database file and related journal files
        if self.db_path != ":memory:":
            p = Path(self.db_path)
            for suffix in ["", "-wal", "-shm", "-journal"]:
                f = p.parent / (p.name + suffix)
                if f.exists():
                    try:
                        f.unlink()
                    except OSError as e:
                        logger.error("Failed to delete %s: %s", f, e)

        # Re-initialize
        self.world = World(filename=self.db_path)
        self.world.graph.db.execute("PRAGMA busy_timeout = 10000")
        self.onto = self.init_world()
        self.seed_axioms()


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

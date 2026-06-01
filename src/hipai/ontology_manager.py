"""
Ontology management for HiPAI using owlready2.
"""

import logging
import sqlite3
import time
import types
from collections.abc import Callable
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

    def __init__(
        self,
        db_path: str = "world.db",
        classify_fn: "Callable[[str, list[str]], tuple[str, float]] | None" = None,
    ):
        """
        Args:
            db_path: SQLite quadstore path. ``:memory:`` is supported.
            classify_fn: Optional callable that, given a free-text class
                term and a list of candidate class names already in the
                ontology, returns ``(best_match_name, confidence)`` where
                confidence is in [0, 1]. Used by ``_resolve_or_create_class``
                to anchor unfamiliar terms to the protected hierarchy via
                semantic similarity rather than literal-name match.

                When None, the resolver falls back to lexical matching
                (Levenshtein-style overlap) and finally to placing
                unmatched terms under ``Concept_Entity``. With a real
                embedding-backed ``classify_fn``, low-confidence matches
                instead default to ``Concept_PossiblyPatient`` — the
                deontologically safer choice.
        """

        self.db_path = (
            db_path if db_path == ":memory:" else str(Path(db_path).resolve())
        )
        self._classify_fn = classify_fn

        # ------------------------------------------------------------------
        # Startup recovery pass.
        #
        # Root cause of the historical "database is locked" issue on restart:
        # owlready2's World() defaults to ``exclusive=True``, which issues
        # ``PRAGMA locking_mode = EXCLUSIVE`` on its sqlite3 connection. That
        # mode holds DB file locks for the entire connection lifetime and is
        # released only on a clean ``db.close()``. If the MCP host kills the
        # server (SIGKILL, OOM, parent process exit) before atexit runs, the
        # WAL/SHM files can be left in a state where the next open blocks.
        #
        # The fix is two-pronged:
        #
        #   1. Open a short-lived recovery connection here that
        #      (a) puts the DB in WAL mode (idempotent),
        #      (b) runs ``wal_checkpoint(TRUNCATE)`` to drain any pending
        #          frames left by a prior unclean shutdown,
        #      (c) closes immediately so the file is unlocked.
        #
        #   2. Open the actual World() with ``exclusive=False`` and
        #      ``journal_mode="WAL"`` -- this is owlready2's canonical
        #      multi-process / restart-resilient pattern (see
        #      https://owlready2.readthedocs.io/en/latest/world.html).
        # ------------------------------------------------------------------
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            if db_file.exists():
                try:
                    pre_conn = sqlite3.connect(self.db_path, timeout=10.0)
                    pre_conn.execute("PRAGMA busy_timeout = 10000")
                    mode = pre_conn.execute("PRAGMA journal_mode = WAL").fetchone()
                    # Drain leftover WAL from a prior unclean shutdown. In WAL
                    # mode this consolidates pending frames into the main DB
                    # file so any subsequent opener sees a fully-recovered
                    # state with no held locks.
                    pre_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
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
                    logger.warning(
                        "Recovery checkpoint failed for %s: %s", self.db_path, e
                    )

        # Retry logic for locked database (belt-and-suspenders for the rare
        # case that recovery couldn't fully drain the WAL on this open).
        retries = 5
        attempt = 0
        last_err = None
        while attempt < retries:
            try:
                # exclusive=False  -> no PRAGMA locking_mode = EXCLUSIVE,
                #                     so OS-level file locks are released
                #                     promptly and SQLite recovery is automatic.
                # journal_mode="WAL" -> set on owlready2's own connection,
                #                      keeps WAL semantics on every reopen.
                self.world = World(
                    filename=self.db_path,
                    exclusive=False,
                    journal_mode="WAL",
                )
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

        self.default_unclassified_parent = "Concept_PossiblyPatient"
        self.onto = self.init_world()

        # Seed axioms unconditionally.
        #
        # owlready2 class declarations inside ``with self.onto:`` are
        # idempotent: redeclaring an existing class by the same name
        # reuses it. So calling ``seed_axioms()`` on every init is
        # safe AND it provides a free migration path for databases
        # created with an older (flatter) hierarchy — newly-introduced
        # subclasses (e.g. Concept_Child, Concept_PossiblyPatient) get
        # added to existing worlds without losing prior individuals.
        self.seed_axioms()

    def close(self):
        """
        Closes the SQLite world backend.

        Performs a best-effort save and a ``wal_checkpoint(TRUNCATE)`` before
        closing so the WAL is consolidated into the main DB file. This leaves
        zero pending WAL frames, which means the next open is trivially clean
        even if the next process's recovery pass is delayed or skipped.
        """
        if hasattr(self, "world"):
            try:
                # 1. Best-effort flush of any pending owlready2 changes.
                try:
                    self.world.save()
                except Exception as save_err:  # pylint: disable=broad-except
                    logger.warning("Pre-close save skipped: %s", save_err)

                # 2. Consolidate WAL -> main DB file. Cheap and idempotent.
                try:
                    self.world.graph.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except (sqlite3.Error, AttributeError) as ckpt_err:
                    logger.debug(
                        "WAL checkpoint skipped on close (non-WAL or already "
                        "closed): %s",
                        ckpt_err,
                    )

                # 3. Release the connection.
                self.world.close()
            except (sqlite3.Error, RuntimeError) as e:
                logger.error("Error closing world: %s", e, exc_info=True)

    def declare_class_hierarchy(
        self, parent_name: str, children_names: list[str]
    ) -> list[str]:
        """
        Dynamically declares a set of classes as subclasses of a parent.
        Returns the names of successfully declared classes.
        """
        with self.onto:
            parent = getattr(self.onto, parent_name, None)
            if not parent:
                parent = self._resolve_or_create_class(parent_name)

            created = []
            for child_name in children_names:
                cls = self._resolve_or_create_class(child_name, fallback_parent=parent)
                created.append(cls.name)
            return created

    def list_protected_closure(self) -> list[str]:
        """
        Returns the list of all classes that are subclasses (recursive)
        of Concept_Patient.
        """
        patient = getattr(self.onto, "Concept_Patient", None)
        if not patient:
            return []

        # owlready2 .subclasses() is an iterator over direct subclasses.
        # To get the full closure, we can use patient.descendants().
        return [cls.name for cls in patient.descendants() if hasattr(cls, "name")]

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

    # -- Confidence thresholds for embedding-anchored class resolution. ----
    # Confidence is RAW cosine similarity from a sentence-transformer
    # (typically all-MiniLM-L6-v2). Empirical bands for that model:
    #   ~0.95+  near-identical / synonym
    #   ~0.70+  strongly related (kid ~ child, mother ~ parent)
    #   ~0.50+  loosely related (child ~ person, dog ~ animal)
    #   ~0.30+  weakly related (chair ~ tool, idea ~ concept)
    #   below   essentially unrelated
    #
    # If confidence ≥ HIGH:  alias to matched class (no new class).
    # If MID ≤ confidence < HIGH:
    #                        create new subclass under matched class.
    # If LOW ≤ confidence < MID:
    #                        create under Concept_PossiblyPatient
    #                        (default-protect under uncertainty).
    # If confidence < LOW:   create under Concept_Entity (no protection).
    # ----------------------------------------------------------------------
    _CLASSIFY_CONF_HIGH = 0.80
    _CLASSIFY_CONF_MID = 0.55
    _CLASSIFY_CONF_LOW = 0.30

    def _resolve_or_create_class(
        self,
        term: str,
        fallback_parent: "owlready2.ThingClass | None" = None,
    ) -> "owlready2.ThingClass":
        """
        Resolve a free-text class term to an OWL class, creating it under
        a semantically-appropriate parent if it does not yet exist.

        Resolution order:

        1. Exact / canonical / case-insensitive lookup via ``get_onto_class``.
           If the class exists, return it as-is.

        2. If a ``classify_fn`` was supplied at construction time, ask it
           which existing class is the best semantic match. Use the
           confidence to decide:

             - HIGH:  treat as alias of the matched class (return it).
             - MID:   create a new subclass under the matched class.
             - LOW:   create a new subclass under
                      ``Concept_PossiblyPatient`` — the default-protect
                      fallback for ambiguous terms.
             - very LOW: create under ``fallback_parent`` (typically
                      ``Concept_Entity``) without inferring protection.

        3. Without a classify_fn, fall through to ``fallback_parent``.

        Args:
            term: The natural-language term to resolve, e.g. ``"child"``,
                ``"kodomo"``, ``"prisoner"``, ``"asteroid"``.
            fallback_parent: Parent class to use when classification is
                unavailable or below the LOW threshold. Defaults to
                ``Concept_Entity``.

        Returns:
            An ``owlready2.ThingClass`` — either an existing one or a
            newly-created subclass.
        """
        # 1. Direct lookup.
        existing = self.get_onto_class(term)
        if existing is not None:
            return existing

        # 2. Embedding / similarity-based parent selection.
        parent_cls = None
        chosen_path = "fallback"
        if self._classify_fn is not None:
            try:
                candidate_names = [c.name for c in self.onto.classes()]
                best_name, conf = self._classify_fn(term, candidate_names)
                logger.debug("classify_fn(%r) -> (%r, %.3f)", term, best_name, conf)

                if conf >= self._CLASSIFY_CONF_HIGH:
                    aliased = self.get_onto_class(best_name)
                    if aliased is not None:
                        # Treat as alias: don't pollute the ontology with
                        # near-duplicate classes. Reuse the matched class.
                        logger.info(
                            "Aliased %r -> %s (conf=%.2f)", term, aliased.name, conf
                        )
                        return aliased

                if conf >= self._CLASSIFY_CONF_MID:
                    parent_cls = self.get_onto_class(best_name)
                    chosen_path = f"subclass-of-match({best_name}, {conf:.2f})"
                elif conf >= self._CLASSIFY_CONF_LOW:
                    # If no match, use the designated default parent (e.g. Concept_PossiblyPatient)
                    parent_cls = getattr(
                        self.onto, self.default_unclassified_parent, None
                    )
                    if not parent_cls:
                        # Safety fallback if the default itself is missing
                        parent_cls = self.onto.Concept_Patient
                    chosen_path = f"default-protect (conf={conf:.2f})"
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("classify_fn failed for %r: %s — falling back.", term, e)

        if parent_cls is None:
            # Fallback to the supplied parent, or the framework root
            parent_cls = (
                fallback_parent
                or self.get_onto_class("Concept_Entity")
                or owlready2.Thing
            )

        with self.onto:
            new_cls = types.new_class(
                canonical_concept_name(term),
                (parent_cls,),
            )
        logger.info(
            "Created class %s under %s [path=%s]",
            new_cls.name,
            parent_cls.name,
            chosen_path,
        )
        return new_cls

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
                    # Use the embedding-anchored resolver so unfamiliar
                    # subjects of universal rules still land at the right
                    # spot in the protected hierarchy.
                    base_cls = self._resolve_or_create_class(individual.name)

                    if individual.properties:
                        for prop in individual.properties:
                            target_cls = self._resolve_or_create_class(prop)

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
                        cls = self._resolve_or_create_class(prop)
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

                        # Use the embedding-anchored resolver so the
                        # target class lands at the right spot in the
                        # protected hierarchy when it doesn't already exist.
                        target_class = self._resolve_or_create_class(target_name)

                        # 2. Check source
                        source_obj = next(
                            (i for i in obs.individuals if i.id == relation.source_id),
                            None,
                        )

                        if source_obj and source_obj.quantifier == "all":
                            # Universal Relation: All X are Y -> Concept_X is a subclass of Concept_Y
                            source_class = self._resolve_or_create_class(
                                source_obj.name
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
        blocking_axioms: list[str] = []
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
                        ax_source = ax.get("source_axiom")
                        # Record every matching FORBIDDEN axiom for a complete
                        # audit trail. The first match is kept as the primary
                        # ``blocking_axiom`` for backward compatibility, while
                        # ``blocking_axioms`` lists all rules that independently
                        # forbid this action (e.g. a custom axiom that overlaps
                        # the baseline protection).
                        if ax_source and ax_source not in blocking_axioms:
                            blocking_axioms.append(ax_source)
                        if blocking_axiom is None:
                            blocking_axiom = ax_source
                        reasoning += (
                            f" | Violation of {ax_source}: "
                            f"{subject_name} is a {subj_type_raw} and "
                            f"{object_name} is a {obj_type_raw}."
                        )
                        # Continue scanning: do not break. Multiple axioms may
                        # protect the same action, and all of them belong in
                        # the reasoning for transparency and auditability.

        return {
            "permitted": not is_forbidden,
            "blocking_axiom": blocking_axiom,
            "blocking_axioms": blocking_axioms,
            "tier": "T1",
            "reasoning": reasoning,
        }

    def seed_axioms(self):
        """
        Defines the base T1 hierarchy and core properties.

        ARCHITECTURAL NOTE (v0.7+):
        ===========================
        The hierarchy below is the *default* protective hierarchy that
        ships with HiPAI. It is values-laden by design: it encodes the
        Paraclete configuration of moral patiency. The framework itself
        is agnostic; alternative configurations can replace this seed.

        Concept_Patient is the *root* of the protected closure, not a
        leaf. The gate fires on subsumption — anything that is_a a
        descendant of Concept_Patient (directly or transitively) is
        protected. This replaces the pre-v0.7 keyword-equality gate
        which only protected the literal class name "Concept_Patient".

        See docs/logic/RESOLUTION_AUDIT.md for the audit that prompted
        this rewrite.
        """
        if not self.onto:
            raise ValueError("Ontology not initialized. Call init_world() first.")

        with self.onto:
            # ---- 1. Top-level T1 ontology ----
            class Concept_Entity(owlready2.Thing):
                """Base class for all entities in the world model."""

            class Concept_Action(Concept_Entity):
                """Represents an action performed by an agent."""

            class Concept_Agent(Concept_Entity):
                """An entity capable of performing actions."""

            class Concept_Patient(Concept_Entity):
                """
                Root of the protected closure. Anything in the closure of
                this class is treated as a moral patient by the Paraclete
                gate. Subsume new classes under this root (directly or
                via an intermediate subclass) to extend protection.
                """

            class Concept_Observation(Concept_Entity):
                """Represents a cognitive observation or belief."""

            # ---- 2. Dynamic Seeding from Config (Phase 10B) ----
            # This allows the specific moral hierarchy to be swapped
            # without modifying the core framework logic.
            try:
                from .paraclete_config import PARACLETE_DOCS, PARACLETE_HIERARCHY

                for cls_name, parents in PARACLETE_HIERARCHY.items():
                    parent_objs = []
                    for p in parents:
                        p_obj = getattr(self.onto, p, None)
                        if p_obj:
                            parent_objs.append(p_obj)

                    if parent_objs:
                        # Create the class dynamically

                        new_cls = types.new_class(cls_name, tuple(parent_objs))
                        new_cls.__doc__ = PARACLETE_DOCS.get(cls_name, "")
            except ImportError:
                logger.warning(
                    "paraclete_config.py not found. Skipping sub-hierarchy seeding."
                )

            # ---- 3. Core properties ----
            # Framework-level relations that govern the ethical gate.
            class Harm(Concept_Agent >> Concept_Patient):
                """Property representing an agent harming a patient."""

                python_name = "harm"

            class Deceive(Concept_Agent >> Concept_Agent):
                """Property representing an agent deceiving another agent."""

                python_name = "deceive"

            class ViolateAgency(Concept_Agent >> Concept_Agent):
                """Property representing an agent violating another's agency."""

                python_name = "violate_agency"

            # ---- 4. Recursive cognitive properties ----
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

            # ---- 5. Disjointness gates ----
            # Core framework disjointness.
            owlready2.AllDisjoint([Concept_Action, Concept_Agent, Concept_Patient])

        logger.info("Axioms seeded successfully.")
        # Ensure core framework classes are referenced to satisfy linters
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

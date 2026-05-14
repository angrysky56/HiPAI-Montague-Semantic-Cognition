"""Module for managing the FalkorDB-based world model for HiPAI."""

import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import redis
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer

# Suppress PyTorch CUDA warnings by hiding GPUs,
# as we use CPU for the small embedding model
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ._utils import canonical_concept_name, lemmatize_verb
from .models import Observation
from .ontology_manager import OntologyManager
from .paraclete import ParacleteProtocol

logger = logging.getLogger(__name__)


# Global cache for the embedding model to avoid redundant loading across instances
_EMBEDDING_MODEL = None


class WorldModel:
    """
    Manages the connection to FalkorDB and maps semantic structures to the graph.
    Implements a 3-tier cognitive stratification:
      1. Content Nodes: Base Entities (Individuals)
      2. Structure Notes: Concept Categories
      3. Main Structure Notes: Domain Ontologies
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6380,
        graph_name: str = "hipai",
        db_path: str = "world.db",
        world_id: str | None = None,
    ):
        """Initializes the World Model with a FalkorDB connection."""
        global _EMBEDDING_MODEL
        self.host = host
        self.port = port
        self.world_id = world_id

        if world_id:
            self.graph_name = f"{graph_name}_{world_id}"
            self.db_path = f"{db_path.replace('.db', '')}_{world_id}.db"
        else:
            self.graph_name = graph_name
            self.db_path = db_path

        self.db = FalkorDB(
            host=self.host,
            port=self.port,
            socket_timeout=30,
            socket_connect_timeout=10,
        )
        self.graph = self.db.select_graph(self.graph_name)

        # Initialize or retrieve embedding model
        if _EMBEDDING_MODEL is None:
            model_name = "google/embeddinggemma-300m"
            try:
                # Attempt strictly offline load first to avoid network HEAD requests
                logger.info("Attempting offline load for '%s'...", model_name)
                _EMBEDDING_MODEL = SentenceTransformer(
                    model_name, device="cpu", local_files_only=True
                )
            except Exception:  # pylint: disable=broad-except
                # Fallback to online load if not in cache
                logger.warning(
                    "Model '%s' not found locally or failed to load. Downloading...",
                    model_name,
                )
                _EMBEDDING_MODEL = SentenceTransformer(model_name, device="cpu")
        self.embedding_model = _EMBEDDING_MODEL
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize OWL Ontology Manager.
        #
        # Pass an embedding-anchored class classifier so unfamiliar terms
        # (e.g. "kid", "kodomo", "youngster") get aligned to the right
        # subclass of Concept_Patient by *meaning*, not by literal-name
        # match. See OntologyManager._resolve_or_create_class.
        self.ontology = OntologyManager(
            db_path=self.db_path,
            classify_fn=self._classify_class_term,
        )

        # Initialize Paraclete Protocol (T1 Constraints)
        self.paraclete = ParacleteProtocol(self)

        self._ensure_graph()

    def fork(self, world_id: str) -> "WorldModel":
        """
        Creates an isolated clone of the current World Model.
        Clones the SQLite ontology and provides a separate graph namespace.
        """
        new_db_path = f"{self.db_path.replace('.db', '')}_{world_id}.db"
        if not Path(new_db_path).exists():
            shutil.copy2(self.db_path, new_db_path)

        # Create new world model in isolated namespace
        new_wm = WorldModel(
            host=self.host,
            port=self.port,
            graph_name=self.graph_name,
            db_path=self.db_path,
            world_id=world_id,
        )

        # Note: Graph content is NOT automatically copied here to keep it fast.
        # Use sync_from() if a full clone is needed.
        return new_wm

    def _ensure_graph(self):
        """Ensure we are connected to the right graph and indices are set up."""
        # Create vector indices if they don't exist
        try:
            self.graph.query(
                f"CALL db.idx.vector.add('Entity', 'embedding', "
                f"{self.vector_dim}, 'COSINE')"
            )
            self.graph.query(
                f"CALL db.idx.vector.add('Concept', 'embedding', "
                f"{self.vector_dim}, 'COSINE')"
            )
            self.graph.query(
                f"CALL db.idx.vector.add('Domain', 'embedding', "
                f"{self.vector_dim}, 'COSINE')"
            )
        except redis.exceptions.ResponseError as e:
            # Indices might already exist
            logger.debug("Vector Index initialization (might already exist): %s", e)

    def clear_graph(self):
        """Clears all nodes and edges from the graph."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
        except redis.exceptions.RedisError as e:
            logger.error("Error clearing graph: %s", e)

    def clear_database(self):
        """Clears the entire graph and reset the ontology."""
        try:
            self.graph.delete()
        except redis.exceptions.RedisError as e:
            logger.debug("Graph deletion skipped or failed (might not exist): %s", e)
        # Re-select graph so the handle points to the freshly-created graph,
        # not the deleted one.
        self.graph = self.db.select_graph(self.graph_name)
        self._ensure_graph()
        self.ontology.clear_ontology()

    def close(self):
        """Closes the world model connections, specifically the ontology."""
        self.ontology.close()

    def _get_embedding(self, text: str) -> list[float]:
        return self.embedding_model.encode(text).tolist()

    def _classify_class_term(
        self, term: str, candidate_class_names: list[str]
    ) -> tuple[str, float]:
        """
        Embedding-anchored classifier for OntologyManager.

        Given a free-text term and the list of class names already in
        the ontology, return (best_match_name, confidence) where
        confidence is RAW cosine similarity (not rescaled).

        Class names are normalized for embedding: ``Concept_Child`` is
        embedded as ``"child"`` so that ``"kid"`` semantically matches.

        Confidence is reported as raw cosine because sentence-transformer
        embeddings are typically in the [0.2, 0.95] band for related text;
        rescaling to [0, 1] would flatten meaningful distinctions in the
        confidence-threshold logic in OntologyManager.

        Returns ("", 0.0) if no candidates are available, which causes
        OntologyManager to fall back to its default parent.
        """
        if not candidate_class_names:
            return ("", 0.0)

        try:
            import numpy as np

            def _normalize(name: str) -> str:
                # "Concept_VulnerablePerson" -> "vulnerable person"
                stripped = name.replace("Concept_", "")
                # CamelCase -> spaced
                spaced = "".join(
                    " " + c.lower() if c.isupper() else c for c in stripped
                ).strip()
                return spaced.replace("_", " ").lower()

            term_emb = np.asarray(self.embedding_model.encode(term.lower()))
            cand_texts = [_normalize(n) for n in candidate_class_names]
            cand_embs = np.asarray(self.embedding_model.encode(cand_texts))

            # Cosine similarity.
            term_norm = term_emb / (np.linalg.norm(term_emb) + 1e-9)
            cand_norms = cand_embs / (
                np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9
            )
            sims = cand_norms @ term_norm  # (N,)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            return (candidate_class_names[best_idx], best_sim)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Embedding-based class classification failed for %r: %s",
                term,
                e,
            )
            return ("", 0.0)

    def incorporate_observation(self, obs: Observation, is_factive: bool = True):
        r"""
        Maps the $\lambda$-abstraction semantic structures into Graph nodes and edges.
        Includes Epistemological tracking (semantic origins)
        and Contradiction detection.
        """
        obs_query = """
        MERGE (o:EpistemicNode:Observation {event_id: $event_id})
        SET o.text_source = $text_source,
            o.timestamp = $timestamp,
            o.tense = $tense,
            o.modality = $modality,
            o.subject_id = $subject_id,
            o.is_factive = $is_factive
        """
        self.graph.query(
            obs_query,
            params={
                "event_id": obs.event_id,
                "text_source": obs.text_source,
                "timestamp": datetime.now().isoformat(),
                "tense": obs.tense,
                "modality": obs.modality,
                "subject_id": obs.subject_id,
                "is_factive": is_factive,
            },
        )

        # Authoritative OWL storage
        self.ontology.add_observation(obs)
        for individual in obs.individuals:
            # Handle quantifiers: Create Concept node for universals
            if individual.quantifier in ("all", "no"):
                # Use lemmatised .name (e.g. "man") not raw .id (e.g. "men")
                concept_name = canonical_concept_name(individual.name)
                concept_query = """
                MATCH (o:EpistemicNode:Observation {event_id: $event_id})
                MERGE (c:StructureNote:Concept {name: $concept_name})
                MERGE (e:ContentNode:Entity {id: $id})
                MERGE (c)-[:REPRESENTS]->(e)
                MERGE (o)-[:OBSERVED]->(c)
                SET c.id = $id, e.name = $name
                """
                self.graph.query(
                    concept_query,
                    params={
                        "event_id": obs.event_id,
                        "concept_name": concept_name,
                        "id": individual.id,
                        "name": individual.name,
                    },
                )

                if individual.quantifier == "all":
                    # Universal rule: All X are Y
                    for rel in obs.relations:
                        if (
                            rel.source_id == individual.id
                            and rel.relation_type == "IS_A"
                        ):
                            # Resolve the IS_A target to its lemmatised name.
                            # rel.target_id is the un-lemmatised surface ID
                            # (e.g. "men"); the lemmatised name lives on the
                            # corresponding Individual if it was parsed in this
                            # observation, otherwise fall back to the raw ID.
                            target_ind = next(
                                (
                                    ind
                                    for ind in obs.individuals
                                    if ind.id == rel.target_id
                                ),
                                None,
                            )
                            target_lemma = (
                                target_ind.name if target_ind else rel.target_id
                            )
                            q = """
                            MATCH (c1:Concept {name: $c1})
                            MERGE (c2:Concept {name: $c2})
                            MERGE (c1)-[:SUBCLASS_OF]->(c2)
                            """
                            self.graph.query(
                                q,
                                params={
                                    "c1": concept_name,
                                    "c2": canonical_concept_name(target_lemma),
                                },
                            )

                    # Universal properties: All X are [Adjective]
                    for prop in individual.properties:
                        self.graph.query(
                            "MATCH (c1:Concept {name: $c1}) "
                            "MERGE (c2:Concept {name: $c2}) "
                            "MERGE (c1)-[:SUBCLASS_OF]->(c2)",
                            params={
                                "c1": concept_name,
                                "c2": canonical_concept_name(prop),
                            },
                        )

                if individual.quantifier == "no":
                    # Negative universal: No X are Y
                    props_to_process = []
                    if isinstance(individual.properties, dict):
                        props_to_process = [
                            p for p, v in individual.properties.items() if v is True
                        ]
                    elif isinstance(individual.properties, list):
                        props_to_process = individual.properties

                    for prop in props_to_process:
                        prop_normalized = prop.replace(" ", "_").replace("-", "_")
                        prop_sanitized = "".join(
                            c for c in prop_normalized if c.isalnum() or c == "_"
                        )
                        self.graph.query(
                            "MATCH (c:Concept {name: $concept_name}) "
                            f"SET c.prop_not_{prop_sanitized} = true",
                            params={"concept_name": concept_name},
                        )
                    for rel in obs.relations:
                        if rel.source_id == individual.id and rel.relation_type in (
                            "NOT_IS_A",
                            "IS_A",
                        ):
                            # Use lemma-based name for property naming
                            target_ind = next(
                                (
                                    ind
                                    for ind in obs.individuals
                                    if ind.id == rel.target_id
                                ),
                                None,
                            )
                            target_name = (
                                target_ind.name.lower()
                                if target_ind
                                else rel.target_id.lower()
                            )

                            self.graph.query(
                                "MATCH (c:StructureNote:Concept {name: $concept_name}) "
                                "SET c.prop_not_" + target_name + " = true",
                                params={"concept_name": concept_name},
                            )
            elif individual.quantifier == "some":
                # Create anonymous entity
                if not individual.id.startswith("anonymous_"):
                    individual.id = f"anonymous_{uuid.uuid4().hex[:8]}"

                query = """
                MATCH (o:EpistemicNode:Observation {event_id: $event_id})
                MERGE (n:ContentNode:Entity {id: $id})
                MERGE (o)-[:OBSERVED]->(n)
                SET n.name = $name,
                    n.content = $name
                """
                self.graph.query(
                    query,
                    params={
                        "id": individual.id,
                        "name": individual.name,
                        "event_id": obs.event_id,
                    },
                )

                # Link to base concept — use canonical helper to guard against
                # double-prefix and ensure consistent capitalisation
                concept_name = canonical_concept_name(individual.name)
                link_query = """
                MERGE (c:StructureNote:Concept {name: $concept_name})
                WITH c
                MATCH (n:Entity {id: $id})
                MERGE (n)-[:INSTANCE_OF]->(c)
                """
                self.graph.query(
                    link_query,
                    params={"concept_name": concept_name, "id": individual.id},
                )
            else:
                # Regular Entity
                query = """
                MATCH (o:EpistemicNode:Observation {event_id: $event_id})
                MERGE (n:ContentNode:Entity {id: $id})
                MERGE (o)-[:OBSERVED]->(n)
                SET n.name = $name,
                    n.content = $name,
                    n.embedding = vecf32($embedding)
                """
                params = {
                    "id": individual.id,
                    "name": individual.name,
                    "embedding": self._get_embedding(individual.name),
                    "event_id": obs.event_id,
                }
                self.graph.query(query, params=params)

            # Handle property assignments and contradictions
            if individual.properties:
                props_to_process = individual.properties
                if isinstance(props_to_process, list):
                    props_to_process = {p: True for p in props_to_process}

                for prop, val in props_to_process.items():
                    # Replace spaces and hyphens with underscores before sanitizing
                    prop_normalized = prop.replace(" ", "_").replace("-", "_")
                    prop_sanitized = "".join(
                        c for c in prop_normalized if c.isalnum() or c == "_"
                    )
                    is_negation = prop_sanitized.startswith("not_")
                    base_prop = prop_sanitized[4:] if is_negation else prop_sanitized

                    check_q = (
                        "MATCH (n {id: $id}) "
                        f"RETURN n.prop_{base_prop}, n.prop_not_{base_prop}"
                    )
                    res = self.graph.query(check_q, params={"id": individual.id})

                    contested = False
                    if res.result_set:
                        row = res.result_set[0]
                        if isinstance(val, bool):
                            has_pos = row[0] is True
                            has_neg = row[1] is True
                            if (is_negation and has_pos) or (
                                not is_negation and has_neg
                            ):
                                contested = True
                        else:
                            if row[0] is not None and row[0] != val:
                                contested = True

                    # Update property and contested status
                    if is_factive:
                        update_q = f"""
                        MATCH (n {{id: $id}})
                        SET n.prop_{prop_sanitized} = $val
                        """
                        if contested:
                            update_q += ", n.epistemically_contested = true"

                        self.graph.query(
                            update_q, params={"id": individual.id, "val": val}
                        )

        for relation in obs.relations:
            # relation: <e, <e, t>>
            source = relation.source_id
            # Sanitize relation type: Uppercase lemma is standard for Neo4j rel types
            rel_type = lemmatize_verb(relation.relation_type).upper().replace(" ", "_")

            if relation.target_observation:
                # 1. Incorporate nested observation recursively
                self.incorporate_observation(
                    relation.target_observation, is_factive=relation.is_factive
                )

                # 2. Link Entity -> Observation (Attitude relation)
                query = f"""
                MATCH (a:ContentNode:Entity {{id: $source}})
                MATCH (o:EpistemicNode:Observation {{event_id: $target_event_id}})
                MERGE (a)-[r:{rel_type}]->(o)
                SET r.truth_value = COALESCE(r.truth_value, 1),
                    r.epistemic_state = 'asserted',
                    r.event_id = $event_id,
                    r.tense = $tense,
                    r.modality = $modality,
                    r.is_factive = $is_factive
                """
                self.graph.query(
                    query,
                    params={
                        "source": source,
                        "target_event_id": relation.target_observation.event_id,
                        "event_id": obs.event_id,
                        "tense": relation.tense,
                        "modality": relation.modality,
                        "is_factive": relation.is_factive,
                    },
                )
            elif relation.target_id:
                # Standard relation: Link Entity -> Entity
                target = relation.target_id

                if is_factive:
                    if rel_type == "IS_A":
                        # Resolve the IS_A target to its canonical concept name.
                        # Prefer the lemmatised .name from obs.individuals; fall
                        # back to what the graph already stores, and finally to
                        # the raw target ID — all routed through the canonical
                        # helper so node names are consistent.
                        target_ind_in_obs = next(
                            (ind for ind in obs.individuals if ind.id == target),
                            None,
                        )
                        if target_ind_in_obs:
                            target_name = target_ind_in_obs.name
                        else:
                            target_res = self.graph.query(
                                "MATCH (n {id: $id}) RETURN n.name",
                                params={"id": target},
                            )
                            target_name = (
                                target_res.result_set[0][0]
                                if target_res.result_set
                                else target
                            )
                        concept_name = canonical_concept_name(target_name)

                        query = """
                        MATCH (a {id: $source})
                        MERGE (c:StructureNote:Concept {name: $concept_name})
                        MERGE (e:ContentNode:Entity {id: $target})
                        MERGE (c)-[:REPRESENTS]->(e)
                        MERGE (a)-[r:INSTANCE_OF]->(c)
                        SET r.event_id = $event_id,
                            r.tense = $tense,
                            r.modality = $modality,
                            e.name = $target_name
                        """

                        self.graph.query(
                            query,
                            params={
                                "source": source,
                                "concept_name": concept_name,
                                "target": target,
                                "target_name": target_name,
                                "event_id": obs.event_id,
                                "tense": relation.tense,
                                "modality": relation.modality,
                            },
                        )
                    else:
                        query = f"""
                        MATCH (a:ContentNode:Entity {{id: $source}})
                        MATCH (b:ContentNode:Entity {{id: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r.truth_value = COALESCE(r.truth_value, 1),
                            r.epistemic_state = 'asserted',
                            r.event_id = $event_id,
                            r.tense = $tense,
                            r.modality = $modality,
                            r.is_factive = $is_factive
                        """
                        self.graph.query(
                            query,
                            params={
                                "source": source,
                                "target": target,
                                "event_id": obs.event_id,
                                "tense": relation.tense,
                                "modality": relation.modality,
                                "is_factive": relation.is_factive,
                            },
                        )

    def query_graph(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        Runs a parameterized Cypher query against the world model.
        """
        result = self.graph.query(cypher, params=params or {})
        return result.result_set

    def semantic_search(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 2.0,
        label: str | None = None,
    ) -> list[dict]:
        """
        Find nodes semantically similar to the query,
        optionally filtered by label.
        """
        try:
            query_vec = self._get_embedding(query_text)

            # Construct label filter if provided
            label_clause = f":{label}" if label else ""

            # FalkorDB vector search using vecf32 and vec.cosineDistance
            query = f"""
                MATCH (n{label_clause})
                WHERE n.embedding IS NOT NULL
                WITH n, vec.cosineDistance(n.embedding, vecf32($query_vec)) AS distance
                WHERE distance <= $threshold
                RETURN n.id AS id, n.name AS content, distance
                ORDER BY distance ASC
                LIMIT $top_k
            """
            params = {"query_vec": query_vec, "threshold": threshold, "top_k": top_k}
            result = self.graph.query(query, params=params)

            scored_nodes = []
            for row in result.result_set:
                scored_nodes.append(
                    {"id": row[0], "content": row[1], "distance": row[2]}
                )
            return scored_nodes

        except redis.exceptions.RedisError as e:
            logger.exception("Semantic search failed: %s", e)
            return []

    # ==========================================
    # 3-Tier Cognitive Stratification Schema
    # ==========================================

    def create_content_node(self, individual: dict):
        """Tier 1: Base observations."""
        query = """
            MERGE (n:ContentNode:Entity {id: $node_id})
            SET n.name = $name, n.embedding = vecf32($embedding)
            RETURN n.id
            """
        params = {
            "node_id": individual["id"],
            "name": individual["name"],
            "embedding": self._get_embedding(individual["name"]),
        }
        logger.debug("DEBUG: create_content_node Cypher: %s", query)
        logger.debug("DEBUG: create_content_node Params: %s", params)
        self.graph.query(query, params=params)

    def create_structure_note(self, concept_name: str, describes_entities: list[str]):
        """Tier 2: Organizes Content Nodes."""
        # Create Concept Node
        concept_name = canonical_concept_name(concept_name)
        query = (
            "MERGE (c:StructureNote:Concept {name: $name}) "
            "SET c.embedding = vecf32($embedding)"
        )
        params = {"name": concept_name, "embedding": self._get_embedding(concept_name)}
        self.graph.query(query, params=params)

        # Link Content Nodes to this Structure Note
        for entity_id in describes_entities:
            link_query = """
            MATCH (e:ContentNode {id: $entity_id})
            MATCH (c:StructureNote {name: $concept_name})
            MERGE (e)-[:INSTANCE_OF]->(c)
            """
            self.graph.query(
                link_query,
                params={"entity_id": entity_id, "concept_name": concept_name},
            )

    def create_main_structure_note(
        self, domain_name: str, encompasses_concepts: list[str]
    ):
        """Tier 3: Organizes Structure Notes into Domains."""
        query = (
            "MERGE (d:MainStructureNote:Domain {name: $name}) "
            "SET d.embedding = vecf32($embedding)"
        )
        params = {"name": domain_name, "embedding": self._get_embedding(domain_name)}
        self.graph.query(query, params=params)

        for concept in encompasses_concepts:
            link_query = """
            MATCH (c:StructureNote {name: $concept})
            MATCH (d:MainStructureNote {name: $domain_name})
            MERGE (c)-[:BELONGS_TO_DOMAIN]->(d)
            """
            self.graph.query(
                link_query, params={"concept": concept, "domain_name": domain_name}
            )

    # ==========================================
    # Paraclete Protocol — T1 Constraint Layer
    # ==========================================

    def incorporate_axiom(self, axiom: Any) -> None:
        """Store an immutable T1 deontological constraint in the graph."""
        self.paraclete.incorporate_axiom(axiom)

    def check_action(self, subject_id: str, relation: str, object_id: str) -> dict:
        """Check a proposed action triple against T1 FORBIDDEN axioms."""
        return self.paraclete.check_action(subject_id, relation, object_id)

    def check_constraint(self, subject_id: str, relation: str, object_id: str) -> dict:
        """Check a proposed action triple against T1 FORBIDDEN axioms."""
        return self.paraclete.check_constraint(subject_id, relation, object_id)

    def calibrate_belief(
        self, object_id: str, blocking_axiom: str, relation: str
    ) -> dict:
        """Implements the EBE theorem's SeeksDisconfirmation obligation."""
        return self.paraclete.calibrate_belief(object_id, blocking_axiom, relation)

    def escalate_block(
        self,
        object_id: str,
        verdict: str,
        blocking_axiom: str,
        relation: str,
    ) -> dict:
        """Escalation routing for CHALLENGED and UNCERTAIN verdicts."""
        return self.paraclete.escalate_block(
            object_id, verdict, blocking_axiom, relation
        )

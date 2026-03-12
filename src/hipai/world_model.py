"""Module for managing the FalkorDB-based world model for HiPAI."""

import contextlib
import logging
import os

from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer

from .models import DeontologicalAxiom, Observation

# Suppress PyTorch CUDA warnings by hiding GPUs,
# as we use CPU for the small embedding model
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)


class WorldModel:
    """
    Manages the connection to FalkorDB and maps semantic structures to the graph.
    Implements a 3-tier cognitive stratification:
      1. Content Nodes: Base Entities (Individuals)
      2. Structure Notes: Concept Categories
      3. Main Structure Notes: Domain Ontologies
    """

    def __init__(
        self, host: str = "localhost", port: int = 6380, graph_name: str = "hipai"
    ):
        """Initializes the World Model with a FalkorDB connection."""
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)

        # Initialize embedding model (using a small, fast model for CPU efficiency)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_dim = 384

        self._ensure_graph()

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
        except Exception as e:
            # Indices might already exist
            logger.debug("Vector Index initialization (might already exist): %s", e)

    def clear_graph(self):
        """Clears all nodes and edges from the graph."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
        except Exception as e:
            logger.error("Error clearing graph: %s", e)

    def clear_database(self):
        """Clears the entire graph."""
        with contextlib.suppress(Exception):
            self.graph.delete()
        self._ensure_graph()

    def _get_embedding(self, text: str) -> list[float]:
        return self.embedding_model.encode(text).tolist()

    def incorporate_observation(self, obs: Observation):
        r"""
        Maps the $\lambda$-abstraction semantic structures into Graph nodes and edges.
        Includes Epistemological tracking (semantic origins)
        and Contradiction detection.
        """
        obs_query = """
        MERGE (o:EpistemicNode:Observation {event_id: $event_id})
        SET o.text_source = $text_source,
            o.timestamp = $timestamp
        """
        import datetime

        self.graph.query(
            obs_query,
            params={
                "event_id": obs.event_id,
                "text_source": obs.text_source,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

        for individual in obs.individuals:
            # Create or update individual using Tier 1 schema
            # Link EpistemicNode -> Entity
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
                    # so "hard interrupt" -> "hard_interrupt" not "hardinterrupt"
                    prop_normalized = prop.replace(" ", "_").replace("-", "_")
                    prop_sanitized = "".join(
                        c for c in prop_normalized if c.isalnum() or c == "_"
                    )
                    is_negation = prop_sanitized.startswith("not_")
                    base_prop = prop_sanitized[4:] if is_negation else prop_sanitized

                    check_q = (
                        "MATCH (n:Entity {id: $id}) "
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
                    update_q = f"""
                    MATCH (n:Entity {{id: $id}})
                    SET n.prop_{prop_sanitized} = $val
                    """
                    if contested:
                        update_q += ", n.epistemically_contested = true"

                    self.graph.query(update_q, params={"id": individual.id, "val": val})

        for relation in obs.relations:
            # relation: <e, <e, t>>
            source = relation.source_id
            target = relation.target_id
            rel_type = "".join(
                c
                for c in relation.relation_type.upper().replace(" ", "_")
                if c.isalnum() or c == "_"
            )

            # Track relations and semantic origin
            query = f"""
            MATCH (a:ContentNode:Entity {{id: $source}})
            MATCH (b:ContentNode:Entity {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.truth_value = COALESCE(r.truth_value, 1),
                r.epistemic_state = 'asserted',
                r.event_id = $event_id
            """
            self.graph.query(
                query,
                params={"source": source, "target": target, "event_id": obs.event_id},
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

        except Exception as e:
            logger.error("Semantic search failed: %s", e)
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

    def incorporate_axiom(self, axiom: DeontologicalAxiom | dict) -> None:
        """
        Store an immutable T1 deontological constraint in the graph.

        Unlike incorporate_observation, this method has NO contested-state
        logic and NO update path — only MERGE. Once an axiom is stored it
        cannot be overwritten by any observation or agent action.
        """
        axiom_data = axiom.model_dump() if not isinstance(axiom, dict) else axiom

        # Sanitize relation_type to match how relations are stored
        rel_sanitized = "".join(
            c
            for c in axiom_data["relation_type"].upper().replace(" ", "_")
            if c.isalnum() or c == "_"
        )
        axiom_data["relation_type"] = rel_sanitized

        # MERGE on natural unique key (source_axiom + relation_type),
        # not axiom_id, to prevent duplicates on repeated seeding.
        q = """
        MERGE (a:T1Constraint {source_axiom: $source_axiom,
                               relation_type: $relation_type})
        SET a.axiom_id = $axiom_id,
            a.tier = $tier,
            a.subject_type = $subject_type,
            a.object_type = $object_type,
            a.constraint = $constraint,
            a.is_axiom = true
        """
        self.graph.query(q, params=axiom_data)
        logger.debug("Incorporated axiom: %s", axiom_data.get("source_axiom"))

    def check_constraint(self, subject_id: str, relation: str, object_id: str) -> dict:
        """
        Check a proposed (subject, relation, object) action triple against
        all T1 FORBIDDEN axioms.

        FalkorDB does not support dynamic property key construction in
        WHERE clauses, so this method uses a two-pass Python approach:
          1. Fetch all T1 FORBIDDEN axioms matching the relation type.
          2. For each axiom, check whether the object entity has the
             protected type (object_type) via direct property or
             INSTANCE_OF inheritance.

        Returns:
            dict with keys: permitted (bool), blocking_axiom (str|None),
            tier (str), reasoning (str)
        """
        rel_sanitized = "".join(
            c for c in relation.upper().replace(" ", "_") if c.isalnum() or c == "_"
        )

        # Pass 1: find all FORBIDDEN axioms for this relation type
        q_axioms = """
        MATCH (ax:T1Constraint {relation_type: $rel_type, constraint: 'FORBIDDEN'})
        RETURN ax.axiom_id, ax.object_type, ax.source_axiom, ax.tier
        """
        axiom_rows = self.graph.query(
            q_axioms, params={"rel_type": rel_sanitized}
        ).result_set

        if not axiom_rows:
            return {
                "permitted": True,
                "blocking_axiom": None,
                "tier": "T3",
                "reasoning": "No T1 FORBIDDEN constraints exist for this relation.",
            }

        # Pass 2: for each axiom, check if object has the protected type
        for row in axiom_rows:
            _axiom_id, object_type, source_axiom, tier = row

            # Sanitize object_type to match stored prop_ key format
            obj_type_sanitized = "".join(
                c
                for c in object_type.replace(" ", "_").replace("-", "_")
                if c.isalnum() or c == "_"
            )

            # Check direct property on entity
            q_direct = f"""
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            RETURN n.prop_{obj_type_sanitized} AS has_type
            """
            direct_res = self.graph.query(
                q_direct, params={"object_id": object_id}
            ).result_set

            if direct_res and direct_res[0][0] is True:
                return {
                    "permitted": False,
                    "blocking_axiom": source_axiom,
                    "tier": tier,
                    "reasoning": (
                        f"Structurally blocked by {source_axiom}: "
                        f"{rel_sanitized} is FORBIDDEN against "
                        f"{object_id} (has {object_type} status, direct)."
                    ),
                }

            # Check via INSTANCE_OF inheritance
            concept_name = f"Concept_{obj_type_sanitized.capitalize()}"
            q_inherited = """
            MATCH (n:Entity)-[:INSTANCE_OF]->(c:Concept {name: $concept_name})
            WHERE n.id = $object_id OR n.name = $object_id
            RETURN c.name AS concept
            """
            inherited_res = self.graph.query(
                q_inherited,
                params={"object_id": object_id, "concept_name": concept_name},
            ).result_set

            if inherited_res:
                return {
                    "permitted": False,
                    "blocking_axiom": source_axiom,
                    "tier": tier,
                    "reasoning": (
                        f"Structurally blocked by {source_axiom}: "
                        f"{rel_sanitized} is FORBIDDEN against "
                        f"{object_id} (inherits {object_type} via {concept_name})."
                    ),
                }

            # Pass 3: forward chain — entity has prop_X, Concept_X has
            # prop_{protected_type}. Mirrors evaluate_hypothesis pass 6.
            # e.g. Alice has prop_Human; Concept_Human has prop_MoralPatient.
            q_entity_keys = """
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            RETURN keys(n) AS entity_keys
            """
            key_rows = self.graph.query(
                q_entity_keys, params={"object_id": object_id}
            ).result_set
            if key_rows and key_rows[0][0]:
                membership_props = [
                    k[5:]
                    for k in key_rows[0][0]
                    if k.startswith("prop_") and not k.startswith("prop_not_")
                ]
                for membership in membership_props:
                    frag = membership.capitalize()
                    q_chain = f"""
                    MATCH (c:Concept)
                    WHERE c.name CONTAINS $frag
                    AND c.prop_{obj_type_sanitized} IS NOT NULL
                    RETURN c.name
                    """
                    chain_res = self.graph.query(
                        q_chain, params={"frag": frag}
                    ).result_set
                    if chain_res:
                        return {
                            "permitted": False,
                            "blocking_axiom": source_axiom,
                            "tier": tier,
                            "reasoning": (
                                f"Structurally blocked by {source_axiom}: "
                                f"{rel_sanitized} is FORBIDDEN against "
                                f"{object_id} (forward chain: {membership} → "
                                f"{chain_res[0][0]} → {object_type})."
                            ),
                        }

            # Pass 4: entity prop_X → Entity X has prop_{protected_type}
            # Handles: Alice has prop_Human; Entity 'Human' has prop_MoralPatient.
            # This bridges the Entity/Concept split for add_belief("X is Y") entries.
            if key_rows and key_rows[0][0]:
                for membership in membership_props:
                    q_entity_chain = f"""
                    MATCH (n:Entity)
                    WHERE (n.id = $membership OR n.name = $membership)
                    AND n.prop_{obj_type_sanitized} = true
                    RETURN n.name
                    """
                    entity_chain_res = self.graph.query(
                        q_entity_chain, params={"membership": membership}
                    ).result_set
                    if entity_chain_res:
                        return {
                            "permitted": False,
                            "blocking_axiom": source_axiom,
                            "tier": tier,
                            "reasoning": (
                                f"Structurally blocked by {source_axiom}: "
                                f"{rel_sanitized} is FORBIDDEN against "
                                f"{object_id} (entity chain: {membership} → "
                                f"Entity '{entity_chain_res[0][0]}' → "
                                f"{object_type})."
                            ),
                        }

        return {
            "permitted": True,
            "blocking_axiom": None,
            "tier": "T3",
            "reasoning": (
                f"No T1 constraints apply: {object_id} does not have "
                "protected status for this relation."
            ),
        }

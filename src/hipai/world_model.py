"""Module for managing the FalkorDB-based world model for HiPAI."""

import contextlib
import logging
import os

from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer

from .models import Observation

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
        Includes Epistemological tracking (semantic origins) and Contradiction detection.
        """
        obs_query = """
        MERGE (o:EpistemicNode:Observation {event_id: $event_id})
        SET o.text_source = $text_source,
            o.timestamp = $timestamp
        """
        import datetime
        self.graph.query(obs_query, params={
            "event_id": obs.event_id,
            "text_source": obs.text_source,
            "timestamp": datetime.datetime.now().isoformat()
        })

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
                "event_id": obs.event_id
            }
            self.graph.query(query, params=params)

            # Handle property assignments and contradictions
            if individual.properties:
                for prop in individual.properties:
                    prop_sanitized = "".join(c for c in prop if c.isalnum() or c == "_")
                    is_negation = prop_sanitized.startswith("not_")
                    base_prop = prop_sanitized[4:] if is_negation else prop_sanitized
                    
                    # Query existing state of the positive and negative properties
                    check_q = f"MATCH (n:Entity {{id: $id}}) RETURN n.prop_{base_prop}, n.prop_not_{base_prop}"
                    res = self.graph.query(check_q, params={"id": individual.id})
                    
                    contested = False
                    if res.result_set:
                        row = res.result_set[0]
                        has_pos = row[0] is True
                        has_neg = row[1] is True
                        if (is_negation and has_pos) or (not is_negation and has_neg):
                            contested = True
                    
                    # Update property and contested status
                    update_q = f"""
                    MATCH (n:Entity {{id: $id}})
                    SET n.prop_{prop_sanitized} = true
                    """
                    if contested:
                        update_q += ", n.epistemically_contested = true"
                    
                    self.graph.query(update_q, params={"id": individual.id})

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
            self.graph.query(query, params={
                "source": source, 
                "target": target, 
                "event_id": obs.event_id
            })

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

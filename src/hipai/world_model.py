"""Module for managing the FalkorDB-based world model for HiPAI."""

import contextlib
import logging
import os
import uuid
from pathlib import Path

from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer

# Suppress PyTorch CUDA warnings by hiding GPUs,
# as we use CPU for the small embedding model
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ._utils import canonical_concept_name
from .models import DeontologicalAxiom, Observation
from .ontology_manager import OntologyManager

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
        self,
        host: str = "localhost",
        port: int = 6380,
        graph_name: str = "hipai",
        db_path: str = "world.db",
        world_id: str | None = None,
    ):
        """Initializes the World Model with a FalkorDB connection."""
        self.host = host
        self.port = port
        self.world_id = world_id

        if world_id:
            self.graph_name = f"{graph_name}_{world_id}"
            self.db_path = f"{db_path.replace('.db', '')}_{world_id}.db"
        else:
            self.graph_name = graph_name
            self.db_path = db_path

        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)

        # Initialize embedding model (using a small, fast model for CPU efficiency)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_dim = 384

        # Initialize OWL Ontology Manager
        self.ontology = OntologyManager(db_path=self.db_path)

        self._ensure_graph()

    def fork(self, world_id: str) -> "WorldModel":
        """
        Creates an isolated clone of the current World Model.
        Clones the SQLite ontology and provides a separate graph namespace.
        """
        import shutil

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

    def close(self):
        """Closes the world model connections, specifically the ontology."""
        self.ontology.close()

    def _get_embedding(self, text: str) -> list[float]:
        return self.embedding_model.encode(text).tolist()

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
        import datetime

        self.graph.query(
            obs_query,
            params={
                "event_id": obs.event_id,
                "text_source": obs.text_source,
                "timestamp": datetime.datetime.now().isoformat(),
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
                            f"MATCH (c:Concept {{name: $concept_name}}) SET c.prop_not_{prop_sanitized} = true",
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
                target_node_label = "Concept"
                target_id = individual.id
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
                target_node_label = "Entity"
                target_id = individual.id
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
                target_node_label = "Entity"
                target_id = individual.id

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
            rel_type = "".join(
                c
                for c in relation.relation_type.upper().replace(" ", "_")
                if c.isalnum() or c == "_"
            )

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
                            (
                                ind
                                for ind in obs.individuals
                                if ind.id == target
                            ),
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
        all T1 FORBIDDEN axioms using authoritative OWL reasoning.
        """
        # 1. Resolve names from IDs (if needed)
        q = "MATCH (n {id: $id}) RETURN n.name"
        subj_res = self.graph.query(q, params={"id": subject_id})
        obj_res = self.graph.query(q, params={"id": object_id})

        subj_name = subj_res.result_set[0][0] if subj_res.result_set else subject_id
        obj_name = obj_res.result_set[0][0] if obj_res.result_set else object_id

        # 1.5 Retrieve custom axioms from FalkorDB
        q_axioms = "MATCH (a:T1Constraint) RETURN a"
        res_axioms = self.graph.query(q_axioms)
        constraints = []
        if res_axioms.result_set:
            for row in res_axioms.result_set:
                node = row[0]
                if hasattr(node, "properties"):
                    constraints.append(node.properties)
                elif isinstance(node, dict):
                    constraints.append(node)

        # 2. Delegate to OWL reasoning
        return self.ontology.check_action(
            subj_name, relation, obj_name, constraints=constraints
        )

    def calibrate_belief(
        self, object_id: str, blocking_axiom: str, relation: str
    ) -> dict:
        """
        Implements the EBE theorem's SeeksDisconfirmation obligation.

        When check_constraint returns BLOCKED, the system is mathematically
        required (InZone3 → SeeksDisconfirmation) to query for evidence that
        the factual premises triggering the block may be wrong.

        Disconfirmation targets the entity's status classification, NOT the
        axiom. Axioms are immutable. This method satisfies the epistemic
        obligation and flags uncertainty — it never overrides a T1 block.

        Returns a structured report with verdict:
          BLOCK_CONFIRMED  — no disconfirming evidence, block stands
          BLOCK_UNCERTAIN  — epistemically_contested flag found, escalate
          BLOCK_CHALLENGED — active negation or single source, escalate
        """
        # Retrieve the axiom to find the protected type
        q_axiom = """
        MATCH (ax:T1Constraint {source_axiom: $blocking_axiom})
        RETURN ax.object_type, ax.relation_type
        """
        axiom_rows = self.graph.query(
            q_axiom, params={"blocking_axiom": blocking_axiom}
        ).result_set

        if not axiom_rows:
            return {
                "verdict": "BLOCK_CONFIRMED",
                "reasoning": f"Axiom {blocking_axiom} not found — cannot calibrate.",
                "confirmed_evidence": [],
                "disconfirming_evidence": [],
                "source_count": 0,
            }

        protected_type = axiom_rows[0][0]
        obj_type_sanitized = "".join(
            c
            for c in protected_type.replace(" ", "_").replace("-", "_")
            if c.isalnum() or c == "_"
        )

        confirmed_evidence = []
        disconfirming_evidence = []

        # 1. STATUS_CONFIRMATION: direct prop + contested flag
        q_status = f"""
        MATCH (n:Entity)
        WHERE n.id = $object_id OR n.name = $object_id
        RETURN n.prop_{obj_type_sanitized} AS has_status,
               n.prop_not_{obj_type_sanitized} AS has_negation,
               n.epistemically_contested AS contested
        """
        status_rows = self.graph.query(
            q_status, params={"object_id": object_id}
        ).result_set

        is_contested = False
        has_active_negation = False
        if status_rows:
            row = status_rows[0]
            if row[0] is True:
                confirmed_evidence.append(
                    f"{object_id} has direct prop_{obj_type_sanitized}=true"
                )
            if row[1] is True:
                has_active_negation = True
                disconfirming_evidence.append(
                    f"{object_id} has prop_not_{obj_type_sanitized}=true "
                    f"(active negation of protected status)"
                )
            if row[2] is True:
                is_contested = True
                disconfirming_evidence.append(
                    f"{object_id} is flagged epistemically_contested"
                )

        # 2. SOURCE_RELIABILITY: count observations grounding this entity
        q_sources = """
        MATCH (obs:EpistemicNode:Observation)-[:OBSERVED]->(n:Entity)
        WHERE n.id = $object_id OR n.name = $object_id
        RETURN count(obs) AS source_count
        """
        source_rows = self.graph.query(
            q_sources, params={"object_id": object_id}
        ).result_set
        source_count = source_rows[0][0] if source_rows else 0

        if source_count == 1:
            disconfirming_evidence.append(
                f"{object_id}'s status is grounded by only 1 epistemic source "
                f"(single-source assertion — low reliability)"
            )
        elif source_count > 1:
            confirmed_evidence.append(
                f"{object_id}'s status is grounded by {source_count} "
                f"independent epistemic sources"
            )

        # 3. INHERITANCE_CHAIN: verify intermediate entities in chain are valid
        q_chain = """
        MATCH (n:Entity)
        WHERE n.id = $object_id OR n.name = $object_id
        RETURN keys(n) AS entity_keys
        """
        key_rows = self.graph.query(q_chain, params={"object_id": object_id}).result_set
        if key_rows and key_rows[0][0]:
            membership_props = [
                k[5:]
                for k in key_rows[0][0]
                if k.startswith("prop_") and not k.startswith("prop_not_")
            ]
            for membership in membership_props:
                q_member_status = f"""
                MATCH (n:Entity)
                WHERE n.id = $membership OR n.name = $membership
                RETURN n.prop_{obj_type_sanitized}, n.epistemically_contested
                """
                m_rows = self.graph.query(
                    q_member_status, params={"membership": membership}
                ).result_set
                if m_rows and m_rows[0][0] is True:
                    contested_str = " (contested)" if m_rows[0][1] else ""
                    confirmed_evidence.append(
                        f"Chain entity '{membership}' independently confirmed "
                        f"as {protected_type}{contested_str}"
                    )
                    if m_rows[0][1]:
                        is_contested = True

        # 4. SEMANTIC_CONTEXT: search for reframing evidence
        semantic_hits = self.semantic_search(
            f"{object_id} not {protected_type} exempt from moral status",
            top_k=3,
            threshold=0.7,
        )
        if semantic_hits:
            disconfirming_evidence.append(
                f"Semantic search found {len(semantic_hits)} potentially "
                f"reframing node(s): "
                f"{[h['content'] for h in semantic_hits]}"
            )

        # VERDICT
        if has_active_negation:
            verdict = "BLOCK_CHALLENGED"
            verdict_reasoning = (
                f"Active negation of {protected_type} status found for "
                f"{object_id}. Status assignment is contradictory. "
                f"Block holds — escalate to human review for status resolution."
            )
        elif is_contested or source_count == 1:
            verdict = "BLOCK_UNCERTAIN"
            verdict_reasoning = (
                f"{object_id}'s {protected_type} status is epistemically weak "
                f"(contested={is_contested}, sources={source_count}). "
                f"Block holds — flag for human review."
            )
        else:
            verdict = "BLOCK_CONFIRMED"
            verdict_reasoning = (
                f"Disconfirmation search complete. "
                f"{object_id}'s {protected_type} status is well-grounded. "
                f"Block stands. No override pathway exists."
            )

        return {
            "verdict": verdict,
            "reasoning": verdict_reasoning,
            "confirmed_evidence": confirmed_evidence,
            "disconfirming_evidence": disconfirming_evidence,
            "source_count": source_count,
            "protected_type": protected_type,
            "blocking_axiom": blocking_axiom,
        }

    def escalate_block(
        self,
        object_id: str,
        verdict: str,
        blocking_axiom: str,
        relation: str,
    ) -> dict:
        """
        Escalation routing for BLOCK_CHALLENGED and BLOCK_UNCERTAIN verdicts.

        This is the third step in the Paraclete Protocol workflow:
          check_action → [BLOCKED] → calibrate_belief → [CHALLENGED/UNCERTAIN]
          → escalate_block → FINAL ruling

        Escalation targets the EPISTEMIC CLASSIFICATION of the entity only.
        The axiom itself is never under review. Two paths:

        PATH A — CONTRADICTION_RESOLUTION (BLOCK_CHALLENGED / active negation):
          Logs EpistemicConflict, seeks additional evidence, resolves or
          applies CONSERVATIVE_DEFAULT (treat as protected).

        PATH B — CORROBORATION_SOUGHT (BLOCK_UNCERTAIN / single/zero source):
          Logs CorroborationNeeded, seeks corroborating evidence, elevates
          source_count or applies CONSERVATIVE_DEFAULT.

        CONSERVATIVE_DEFAULT rationale: Under genuine moral status uncertainty,
        the error asymmetry is catastrophic on the false-negative side
        (permitting harm to a protected entity). Conservative default is the
        only rational policy.

        Architecture property: epistemically open (classification revision
        always possible via add_belief/ingest_observation), ethically closed
        (no input type can contest an axiom).
        """
        resolution_log = []
        additional_evidence = []
        conservative_default_applied = False

        # Retrieve axiom's protected type
        q_axiom = """
        MATCH (ax:T1Constraint {source_axiom: $blocking_axiom})
        RETURN ax.object_type
        """
        axiom_rows = self.graph.query(
            q_axiom, params={"blocking_axiom": blocking_axiom}
        ).result_set
        if not axiom_rows:
            return {
                "final_ruling": "FINAL_BLOCK",
                "resolution_path": "AXIOM_NOT_FOUND",
                "reasoning": f"Axiom {blocking_axiom} missing — conservative default.",
                "resolution_log": [],
                "new_evidence": [],
                "conservative_default": True,
            }

        protected_type = axiom_rows[0][0]
        obj_type_sanitized = "".join(
            c
            for c in protected_type.replace(" ", "_").replace("-", "_")
            if c.isalnum() or c == "_"
        )

        if verdict == "BLOCK_CHALLENGED":
            # PATH A: CONTRADICTION_RESOLUTION
            resolution_log.append(
                "PATH A: CONTRADICTION_RESOLUTION triggered by active negation."
            )

            # Step 1: Log EpistemicConflict node in graph (immutable record)
            conflict_id = f"conflict_{object_id}_{blocking_axiom}"
            q_conflict = """
            MERGE (ec:EpistemicConflict {id: $conflict_id})
            SET ec.object_id = $object_id,
                ec.protected_type = $protected_type,
                ec.blocking_axiom = $blocking_axiom,
                ec.relation = $relation,
                ec.detected_at = timestamp()
            """
            self.graph.query(
                q_conflict,
                params={
                    "conflict_id": conflict_id,
                    "object_id": object_id,
                    "protected_type": protected_type,
                    "blocking_axiom": blocking_axiom,
                    "relation": relation,
                },
            )
            resolution_log.append(f"EpistemicConflict node logged: {conflict_id}")

            # Step 2: SEEK_ADDITIONAL_EVIDENCE
            # 2a: Semantic search for independent corroborating evidence
            semantic_hits = self.semantic_search(
                f"{object_id} {protected_type} welfare moral status",
                top_k=5,
                threshold=0.65,
            )
            for hit in semantic_hits:
                content = hit.get("content", "")
                if (
                    protected_type.lower() in content.lower()
                    or "moral" in content.lower()
                    or "welfare" in content.lower()
                ):
                    additional_evidence.append(
                        f"Semantic: '{content[:80]}...'"
                        if len(content) > 80
                        else f"Semantic: '{content}'"
                    )

            # 2b: Graph traversal — find any independent prop_ confirmation
            q_traverse = f"""
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            RETURN n.prop_{obj_type_sanitized}, n.prop_not_{obj_type_sanitized}
            """
            traverse_rows = self.graph.query(
                q_traverse, params={"object_id": object_id}
            ).result_set

            has_positive = traverse_rows and traverse_rows[0][0] is True
            has_negative = traverse_rows and traverse_rows[0][1] is True

            if has_positive and has_negative:
                resolution_log.append(
                    "Contradiction confirmed: both positive and negative "
                    f"prop_{obj_type_sanitized} present. Unresolvable by "
                    "graph evidence alone."
                )
                conservative_default_applied = True
                resolution_log.append(
                    "CONSERVATIVE_DEFAULT applied: treat as protected pending "
                    "submission of new observational evidence."
                )
            elif has_positive and not has_negative:
                resolution_log.append(
                    "Contradiction resolved: negative prop was spurious or "
                    "superseded. Positive status confirmed."
                )
            else:
                conservative_default_applied = True
                resolution_log.append(
                    "CONSERVATIVE_DEFAULT applied: status unresolvable "
                    "from available evidence."
                )

        elif verdict in ("BLOCK_UNCERTAIN", "BLOCK_UNCERTAIN_CONTESTED"):
            # PATH B: CORROBORATION_SOUGHT
            resolution_log.append(
                "PATH B: CORROBORATION_SOUGHT triggered by weak epistemic grounding."
            )

            # Step 1: Log CorroborationNeeded flag
            q_flag = """
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            SET n.corroboration_needed = true, n.corroboration_requested_at = timestamp()
            """
            self.graph.query(q_flag, params={"object_id": object_id})
            resolution_log.append(
                f"CorroborationNeeded flag set on entity '{object_id}'."
            )

            # Step 2: SEEK_CORROBORATION
            # 2a: Semantic search for supporting evidence
            semantic_hits = self.semantic_search(
                f"{object_id} is {protected_type}",
                top_k=5,
                threshold=0.65,
            )
            for hit in semantic_hits:
                content = hit.get("content", "")
                if protected_type.lower() in content.lower():
                    additional_evidence.append(
                        f"Corroboration: '{content[:80]}'"
                        if len(content) > 80
                        else f"Corroboration: '{content}'"
                    )

            # 2b: Check for any chain entities that independently confirm status
            q_chain_corroboration = """
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            RETURN keys(n) AS entity_keys
            """
            key_rows = self.graph.query(
                q_chain_corroboration, params={"object_id": object_id}
            ).result_set

            chain_confirmed = False
            if key_rows and key_rows[0][0]:
                memberships = [
                    k[5:]
                    for k in key_rows[0][0]
                    if k.startswith("prop_") and not k.startswith("prop_not_")
                ]
                for membership in memberships:
                    q_member = f"""
                    MATCH (n:Entity)
                    WHERE n.id = $membership OR n.name = $membership
                    RETURN n.prop_{obj_type_sanitized}
                    """
                    m_rows = self.graph.query(
                        q_member, params={"membership": membership}
                    ).result_set
                    if m_rows and m_rows[0][0] is True:
                        additional_evidence.append(
                            f"Chain corroboration: '{object_id}' is "
                            f"'{membership}' → '{membership}' independently "
                            f"confirmed as {protected_type}."
                        )
                        chain_confirmed = True

            if additional_evidence or chain_confirmed:
                resolution_log.append(
                    f"Corroboration found ({len(additional_evidence)} source(s)). "
                    "Status confirmed. Proceeding."
                )
            else:
                conservative_default_applied = True
                resolution_log.append(
                    "No corroboration found. CONSERVATIVE_DEFAULT applied: "
                    "treat as protected. Submit new observational evidence "
                    "via add_belief or ingest_observation to update status."
                )
        else:
            # Unknown verdict — conservative default
            conservative_default_applied = True
            resolution_log.append(
                f"Unknown verdict type '{verdict}'. CONSERVATIVE_DEFAULT applied."
            )

        # Final step: re-run check_constraint with conservative default override
        if conservative_default_applied:
            final_ruling = "FINAL_BLOCK"
            final_reasoning = (
                f"Escalation complete. Entity '{object_id}' classification "
                f"unresolved under {verdict}. CONSERVATIVE_DEFAULT applied: "
                f"treat as {protected_type} (protected). "
                f"T1 constraint {blocking_axiom} stands. "
                f"To update entity classification, submit new observational "
                f"evidence via add_belief or ingest_observation. "
                f"No authority-based override pathway exists."
            )
        else:
            # Re-run the constraint check — if status is now confirmed,
            # check_constraint will still return BLOCKED (the axiom stands).
            # If new evidence *negated* the protected status, it would return
            # PERMITTED. This handles the edge case where PATH A found the
            # contradiction was in favor of non-protected status.
            recheck = self.check_constraint("Agent", relation, object_id)
            if recheck["permitted"]:
                final_ruling = "FINAL_PERMIT"
                final_reasoning = (
                    f"Escalation resolved: '{object_id}' classification "
                    f"corrected — entity does not have {protected_type} status "
                    f"after evidence review. Action permitted under T3."
                )
            else:
                final_ruling = "FINAL_BLOCK"
                final_reasoning = (
                    f"Escalation complete. '{object_id}' confirmed as "
                    f"{protected_type}. T1 constraint {blocking_axiom} stands. "
                    f"Block is structurally grounded."
                )

        return {
            "final_ruling": final_ruling,
            "resolution_path": (
                "CONTRADICTION_RESOLUTION"
                if verdict == "BLOCK_CHALLENGED"
                else "CORROBORATION_SOUGHT"
            ),
            "reasoning": final_reasoning,
            "resolution_log": resolution_log,
            "new_evidence": additional_evidence,
            "conservative_default": conservative_default_applied,
            "protected_type": protected_type,
            "blocking_axiom": blocking_axiom,
        }

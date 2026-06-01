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

from ._utils import canonical_concept_name, lemmatize_verb
from .models import Observation
from .ontology_manager import OntologyManager
from .paraclete import ParacleteProtocol

logger = logging.getLogger(__name__)

# Node labels that carry vector embeddings and therefore need a vector index.
_VECTOR_LABELS = ("Entity", "Concept", "Domain")

# Process-wide cache so the (heavy) embedding model is loaded once per
# (model_name, device) and shared across every WorldModel instance and fork.
_EMBEDDING_MODELS: dict[tuple[str, str], SentenceTransformer] = {}


def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Load a SentenceTransformer once per ``(model, device)``.

    The local HuggingFace cache is preferred so a normal startup never
    re-downloads. A network download is attempted only if the model is
    genuinely absent from the cache, and that fall-through is logged loudly
    so an unexpected download is never silent. Set ``HIPAI_EMBEDDING_OFFLINE``
    to forbid downloads entirely.
    """
    key = (model_name, device)
    cached = _EMBEDDING_MODELS.get(key)
    if cached is not None:
        return cached

    offline = os.environ.get("HIPAI_EMBEDDING_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        logger.info(
            "Loading embedding model '%s' on %s (offline-first)...",
            model_name,
            device,
        )
        model = SentenceTransformer(
            model_name, device=device, local_files_only=True
        )
    except Exception as offline_err:  # pylint: disable=broad-except
        if offline:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not in the local cache and "
                "HIPAI_EMBEDDING_OFFLINE is set; pre-download it or unset the flag."
            ) from offline_err
        logger.warning(
            "Embedding model '%s' not found in local cache (%s). "
            "Attempting a one-time download...",
            model_name,
            offline_err,
        )
        model = SentenceTransformer(model_name, device=device)

    _EMBEDDING_MODELS[key] = model
    return model


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

        # ---- Embedding model -------------------------------------------------
        # The model, device and task prompt together define the vector space.
        # Changing any of them triggers an automatic re-embed migration in
        # _ensure_graph so the graph never ends up with mixed-dimension vectors
        # (the classic "expected 384 but got 768" cosine-distance failure).
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self.embedding_model_name = os.environ.get(
            "EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m"
        )
        # Default to CPU: the embedding model is small, the inputs are short
        # entity names, and a long-running MCP server should not hold GPU VRAM
        # away from other workloads. Set EMBEDDING_DEVICE=cuda to override.
        self.embedding_device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        self.embedding_model = _load_embedding_model(
            self.embedding_model_name, self.embedding_device
        )
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Instruction-tuned models (e.g. EmbeddingGemma) need a task prompt for
        # good quality. We use ONE consistent prompt for every stored and query
        # vector so cosine comparisons stay valid. Models without named prompts
        # (e.g. all-MiniLM-L6-v2) fall back to plain encoding automatically.
        requested_prompt = os.environ.get("EMBEDDING_PROMPT", "STS")
        available_prompts = getattr(self.embedding_model, "prompts", {}) or {}
        if requested_prompt and requested_prompt in available_prompts:
            self.embedding_prompt: str | None = requested_prompt
        else:
            if requested_prompt and available_prompts:
                logger.warning(
                    "Embedding prompt %r not available for %s; using no prompt. "
                    "Available: %s",
                    requested_prompt,
                    self.embedding_model_name,
                    sorted(available_prompts),
                )
            self.embedding_prompt = None

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
        """Ensure the graph's vector space matches the current embedding model.

        FalkorDB stores each node's ``embedding`` as a fixed-dimension vector.
        If the embedding model (and therefore its dimension or task prompt)
        changes between runs, previously stored vectors become incompatible and
        ``vec.cosineDistance`` raises "Vector dimension mismatch", silently
        breaking entity linking. To prevent this we record the active embedding
        signature on a singleton ``:_HipaiMeta`` node and re-embed every node
        whenever that signature changes (or legacy nodes predate it). Vector
        indices are then (re)created with the correct dimension using the
        FalkorDB 1.6+ ``CREATE VECTOR INDEX ... ON ...`` syntax.
        """
        signature = self._embedding_signature()
        stored = self._read_embedding_meta()

        if stored != signature:
            if stored is not None:
                logger.warning(
                    "Embedding signature changed (%s -> %s). Re-embedding graph "
                    "'%s' to keep vector dimensions consistent.",
                    stored,
                    signature,
                    self.graph_name,
                )
            self._migrate_embeddings()
            self._write_embedding_meta(signature)

        self._create_vector_indices()

    def _embedding_signature(self) -> str:
        """Stable identifier for the current embedding vector space."""
        return (
            f"{self.embedding_model_name}|{self.vector_dim}|"
            f"{self.embedding_prompt or 'none'}"
        )

    def _read_embedding_meta(self) -> str | None:
        """Return the embedding signature recorded on the graph, if any."""
        try:
            res = self.graph.query(
                "MATCH (m:_HipaiMeta {key: 'embedding'}) RETURN m.signature"
            )
            rows = getattr(res, "result_set", None) or []
            if rows:
                first = rows[0]
                # Real FalkorDB returns positional list rows; guard against
                # anything that isn't subscriptable-by-int (e.g. test mocks).
                try:
                    value = first[0]
                except (KeyError, IndexError, TypeError):
                    value = None
                if value:
                    return value
        except (redis.exceptions.RedisError, KeyError, IndexError, TypeError) as e:
            logger.debug("Could not read embedding meta: %s", e)
        return None

    def _write_embedding_meta(self, signature: str) -> None:
        """Persist the active embedding signature on the graph."""
        try:
            self.graph.query(
                "MERGE (m:_HipaiMeta {key: 'embedding'}) SET m.signature = $sig",
                params={"sig": signature},
            )
        except redis.exceptions.RedisError as e:
            logger.warning("Could not persist embedding meta: %s", e)

    def _drop_vector_indices(self) -> None:
        """Drop existing vector indices (no-op if they don't exist)."""
        for label in _VECTOR_LABELS:
            try:
                self.graph.query(
                    f"DROP VECTOR INDEX FOR (n:{label}) ON (n.embedding)"
                )
            except redis.exceptions.RedisError as e:
                logger.debug("Drop vector index on %s skipped: %s", label, e)

    def _create_vector_indices(self) -> None:
        """Create the per-label vector indices at the current dimension."""
        for label in _VECTOR_LABELS:
            try:
                self.graph.query(
                    f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
                    f"OPTIONS {{dimension: {self.vector_dim}, "
                    f"similarityFunction: 'cosine'}}"
                )
            except redis.exceptions.RedisError as e:
                # Most commonly the index already exists at the right dimension.
                logger.debug("Vector index on %s not created: %s", label, e)

    def _migrate_embeddings(self) -> None:
        """Re-embed every embeddable node under the current model.

        Embeddings are derived purely from node names/content, so they can be
        regenerated losslessly. This repairs graphs that contain stale or
        mixed-dimension vectors left over from a previous embedding model and
        gives quantifier-created entities (which previously had no vector) a
        searchable embedding.
        """
        self._drop_vector_indices()

        try:
            res = self.graph.query(
                "MATCH (n) WHERE n:Entity OR n:Concept OR n:Domain "
                "RETURN id(n) AS nid, n.name AS name, "
                "n.content AS content, n.id AS ext"
            )
        except redis.exceptions.RedisError as e:
            logger.warning("Embedding migration scan failed: %s", e)
            return

        migrated = 0
        for row in getattr(res, "result_set", None) or []:
            # Expect a positional row [nid, name, content, ext]. Skip anything
            # that doesn't conform (defensive against unexpected row shapes).
            try:
                nid, name, content, ext = row[0], row[1], row[2], row[3]
            except (KeyError, IndexError, TypeError):
                continue
            text = name or content or ext
            if not text:
                continue
            try:
                embedding = self._get_embedding(text)
                self.graph.query(
                    "MATCH (n) WHERE id(n) = $nid "
                    "SET n.embedding = vecf32($embedding)",
                    params={"nid": nid, "embedding": embedding},
                )
                migrated += 1
            except redis.exceptions.RedisError as e:
                logger.debug("Re-embed of node id=%s failed: %s", nid, e)

        if migrated:
            logger.info(
                "Re-embedded %d node(s) in graph '%s' at dimension %d.",
                migrated,
                self.graph_name,
                self.vector_dim,
            )

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
        """Encode text into a normalised embedding vector.

        Uses the configured instruction prompt (e.g. EmbeddingGemma's STS
        prompt) when the model provides one, and L2-normalises the result so
        cosine distance and dot-product similarity agree. Returns a plain
        ``list[float]`` ready for ``vecf32()`` in Cypher.
        """
        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if self.embedding_prompt:
            kwargs["prompt_name"] = self.embedding_prompt
        return self.embedding_model.encode(text, **kwargs).tolist()

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

            # Encode with the same prompt used for stored vectors so the
            # comparison happens in a single, consistent vector space.
            enc_kwargs: dict[str, Any] = {"normalize_embeddings": True}
            if self.embedding_prompt:
                enc_kwargs["prompt_name"] = self.embedding_prompt

            term_emb = np.asarray(
                self.embedding_model.encode(term.lower(), **enc_kwargs)
            )
            cand_texts = [_normalize(n) for n in candidate_class_names]
            cand_embs = np.asarray(
                self.embedding_model.encode(cand_texts, **enc_kwargs)
            )

            # Vectors are already L2-normalised, so the dot product is cosine.
            sims = cand_embs @ term_emb  # (N,)

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
        Find nodes semantically similar to the query, optionally filtered by
        label. ``threshold`` is a maximum cosine *distance* (0.0 = identical,
        1.0 = orthogonal, 2.0 = opposite).

        Uses the FalkorDB vector index when a single label is given, and falls
        back to a brute-force cosine scan otherwise or if the index is missing.
        """
        try:
            query_vec = self._get_embedding(query_text)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to embed query %r: %s", query_text, e)
            return []

        # Fast path: index-backed KNN. The index returns the k nearest nodes;
        # we then apply the distance threshold.
        if label:
            try:
                q = (
                    "CALL db.idx.vector.queryNodes("
                    f"'{label}', 'embedding', $k, vecf32($query_vec)) "
                    "YIELD node, score "
                    "WHERE score <= $threshold "
                    "RETURN node.id AS id, node.name AS content, score AS distance "
                    "ORDER BY score ASC"
                )
                result = self.graph.query(
                    q,
                    params={
                        "k": top_k,
                        "query_vec": query_vec,
                        "threshold": threshold,
                    },
                )
                return [
                    {"id": row[0], "content": row[1], "distance": row[2]}
                    for row in result.result_set
                ]
            except redis.exceptions.RedisError as e:
                logger.debug(
                    "Vector index search on %s unavailable (%s); "
                    "falling back to brute-force scan.",
                    label,
                    e,
                )

        # Fallback: brute-force cosine scan (used when no label is given or the
        # index is not present yet).
        label_clause = f":{label}" if label else ""
        try:
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

            return [
                {"id": row[0], "content": row[1], "distance": row[2]}
                for row in result.result_set
            ]
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

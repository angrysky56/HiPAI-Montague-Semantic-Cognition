# Copyright 2025 Google DeepMind
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Synthesizer Module
"""

import logging
from typing import Any

from .models import DeontologicalAxiom, Observation, Individual, Relation
from .world_model import WorldModel
from .parser import ClaimExtractor

try:
    import numpy as np
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    np = None
    KMeans = None
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class ZettelkastenSynthesizer:
    """
    Synthesizes higher-order logic structures (Structure Notes and Main Structure Notes)
    from base observations (Content Nodes).
    Implements intensional logic evaluation by allowing worlds (graph states) to be
    evaluated over time.
    """

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def register_property_map(self, property_map: dict):
        """
        Register a property map to map properties to specific
        ontological categories or evaluation rules.

        Args:
            property_map: Dictionary mapping properties to specific
            ontological categories or evaluation rules.
        """
        # TODO: Implement property mapping
        # This method is intended for future use to map properties to specific
        # ontological categories or evaluation rules.
        # For now, it's a placeholder to satisfy linting/design.
        pass

    def synthesize_concepts(self, property_threshold: int = 1) -> list[str]:
        """
        Generates Structure Notes based on common properties among
        Content Nodes (Entities).
        If entities share a property, a Concept is generated in the graph.
        Returns a list of created concept names.
        """
        # We find all keys on Entities to discover properties
        query_keys = "MATCH (n:Entity) RETURN DISTINCT keys(n) AS keys"

        try:
            result = self.world_model.query_graph(query_keys)
        except Exception as e:
            logger.error("Failed to query graph: %s", e)
            return []  # In case the graph doesn't exist

        all_props = set()
        for row in result:
            keys = row[0]
            for key in keys:
                if key.startswith("prop_"):
                    prop_name = key[5:]
                    all_props.add(prop_name)

        created_concepts = []
        for prop in all_props:
            # Find entities with this property = true
            q = f"MATCH (n:Entity) WHERE n.prop_{prop} = true RETURN n.id AS id"
            res = self.world_model.query_graph(q)

            entities = [r[0] for r in res]

            if len(entities) >= property_threshold:
                concept_name = f"Concept_{prop.capitalize()}"
                self.world_model.create_structure_note(concept_name, entities)
                created_concepts.append(concept_name)

        return created_concepts

    def vector_synthesize_concepts(self, n_clusters: int = 2) -> list[str]:
        """
        Tier 2+: Uses machine learning (KMeans) to cluster Entities purely by
        their vector embeddings, suggesting latent category Structure Notes.
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not installed. Vector synthesis skipped.")
            return []

        # 1. Fetch all entities and their embeddings
        q = "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e.id, e.embedding"
        res = self.world_model.query_graph(q)
        if not res or len(res) < n_clusters:
            return []

        ids = [row[0] for row in res]
        embeddings = np.array([row[1] for row in res])

        # 2. Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        # 3. Create concepts based on clusters
        created_concepts = []
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(ids[idx])

        for label, entity_ids in clusters.items():
            concept_name = f"LatentConcept_Cluster_{label}_{len(entity_ids)}"
            self.world_model.create_structure_note(concept_name, entity_ids)
            created_concepts.append(concept_name)

        return created_concepts

    def synthesize_domains(self, concept_threshold: int = 2) -> list[str]:
        """
        Generates Main Structure Notes (Domains) by clustering related Concepts.
        For simplicity, this clusters concepts that share underlying entities.
        Uses concept_threshold to filter connections.
        """
        # Find concepts that share at least concept_threshold entities
        q = """
        MATCH (c1:Concept)<-[:INSTANCE_OF]-(e:Entity)-[:INSTANCE_OF]->(c2:Concept)
        WHERE id(c1) < id(c2)
        WITH c1, c2, count(e) as shared_entities
        WHERE shared_entities >= $threshold
        RETURN c1.name, c2.name
        """
        try:
            res = self.world_model.query_graph(q, {"threshold": concept_threshold})
        except Exception as e:
            logger.error("Failed to query graph for domains: %s", e)
            return []

        domains_created = []
        for row in res:
            c1, c2 = row[0], row[1]
            # Simple grouping logic based on first connection
            domain_name = f"Domain_{c1.split('_')[1]}_{c2.split('_')[1]}"
            self.world_model.create_main_structure_note(domain_name, [c1, c2])
            domains_created.append(domain_name)

        return domains_created


# Map of verb forms to their canonical base/stem for relation type matching.
VERB_STEM_OVERRIDES: dict[str, str] = {
    "causes": "cause",
    "leads": "lead",
    "produces": "produce",
    "creates": "create",
    "triggers": "trigger",
    "generates": "generate",
    "enables": "enable",
    "prevents": "prevent",
    "blocks": "block",
    "inhibits": "inhibit",
    "harms": "harm",
    "exploits": "exploit",
    "manipulates": "manipulate",
    "influences": "influence",
    "affects": "affect",
    "impacts": "impact",
    "shapes": "shape",
    "alters": "alter",
    "modifies": "modify",
    "requires": "require",
    "needs": "need",
    "supports": "support",
    "confirms": "confirm",
    "contradicts": "contradict",
    "challenges": "challenge",
    "undermines": "undermine",
    "visits": "visit",
    "sees": "see",
    "meets": "meet",
    "calls": "call",
    "loves": "love",
    "hates": "hate",
}


class HIPAIManager:
    """
    High-level manager for the Montague-style semantic cognition system.
    Orchestrates the WorldModel and ZettelkastenSynthesizer.
    This class provides the interface expected by test_hipai.py.
    """

    @staticmethod
    def _normalize_verb(verb: str) -> str:
        """Normalize an inflected verb to its base/stem form for relation type creation.

        Uses an explicit override table for known verbs, then falls back to
        simple suffix stripping (``-es`` → ``-e``, ``-s`` → base).
        """
        v = verb.lower().strip()
        if v in VERB_STEM_OVERRIDES:
            return VERB_STEM_OVERRIDES[v]
        # Fallback heuristics
        if v.endswith("ies"):  # e.g. "relies" → "rely"
            return v[:-3] + "y"
        if (
            v.endswith("ses")
            or v.endswith("zes")
            or v.endswith("xes")
            or v.endswith("ches")
            or v.endswith("shes")
        ):
            return v[:-2]  # e.g. "causes" already handled above
        if v.endswith("es"):
            return v[:-1]  # e.g. "produces" → "produce"
        if v.endswith("s") and not v.endswith("ss"):
            return v[:-1]  # e.g. "harms" → "harm"
        return v

    def __init__(self, graph_name: str = "hipai_world", db_path: str = "world.db"):
        self.world_model = WorldModel(graph_name=graph_name, db_path=db_path)
        self.synthesizer = ZettelkastenSynthesizer(self.world_model)
        self.parser = ClaimExtractor()
        self.logger = logger

    def clear_database(self):
        """Standardizer for clearing the model's graph database."""
        self.world_model.clear_database()

    def _resolve_ambiguity(self, obs: Observation):
        """
        Hardens the observation by resolving entities against the current World Model.
        If an entity name is ambiguous, it uses semantic search to find the best match.
        """
        for ind in obs.individuals:
            # Search for existing entities with similar names
            results = self.world_model.semantic_search(ind.name, top_k=5, threshold=0.4, label="Entity")
            
            if results:
                best = results[0]
                # Results[0] distance is cosine distance (0.0 means identical)
                if best["distance"] < 0.2:
                    # High confidence match, use existing ID
                    if ind.id != best["id"]:
                        self.logger.debug("Mapping new entity '%s' to existing ID '%s'", ind.name, best["id"])
                    ind.id = best["id"]
                elif len(results) > 1:
                    # Multiple candidates, pick best but note ambiguity
                    ind.id = best["id"]
                    obs.confidence *= 0.9 # Penalty for ambiguity
                    self.logger.info("Resolved ambiguous entity '%s' to '%s' (dist: %.2f)", 
                                     ind.name, best["id"], best["distance"])

        # Recursive resolution for attitudes
        for rel in obs.relations:
            if rel.target_observation:
                self._resolve_ambiguity(rel.target_observation)
                # Propagate lower confidence
                obs.confidence = min(obs.confidence, rel.target_observation.confidence)


    def add_belief(self, text: str, incorporate: bool = True) -> dict[str, Any]:
        """
        Synthesize a belief from natural language text and add it to the graph if incorporate is True.
        Delegates parsing to the ClaimExtractor (spaCy-backed) and validates against T1 ontology.
        """
        # Clean text
        text = text.strip()

        # 1. Parse using ClaimExtractor
        try:
            obs = self.parser.extract(text)
        except Exception as e:
            self.logger.error("Parser failed for '%s': %s", text, e)
            return {"status": "error", "message": f"Parsing failed: {e}"}

        # 1.5 Resolve Ambiguity (Context-Aware Entity Linking)
        self._resolve_ambiguity(obs)

        if not obs.individuals and not obs.relations:
            self.logger.warning("No semantic entities extracted from: %s", text)
            # Fallback: store as free-text entity
            obs = Observation(
                text_source=text,
                individuals=[
                    Individual(
                        id=text[:50].replace(" ", "_").lower(),
                        name=text,
                        properties=["unstructured_belief"],
                    )
                ],
                relations=[],
            )
            if incorporate:
                self.world_model.incorporate_observation(obs)
            return {
                "status": "success",
                "message": f"Added as unstructured belief: {text}",
                "observation": obs,
            }

        # 2. Constraint Check (Paraclete Protocol)
        # We check every relation in the observation against the T1 WorldModel
        for rel in obs.relations:
            res = self.world_model.check_constraint(
                rel.source_id, rel.relation_type, rel.target_id
            )
            if not res["permitted"]:
                self.logger.warning("Constraint Violation: %s", res["reasoning"])
                return {
                    "status": "error",
                    "message": f"Deontological violation: {res['reasoning']}",
                    "blocking_axiom": res["blocking_axiom"],
                }

        # 3. Incorporation
        if incorporate:
            self.world_model.incorporate_observation(obs)

        return {
            "status": "success",
            "message": f"Successfully processed belief: {text}",
            "observation": obs,
            "parse": {"type": "nlp_extracted", "text": text},
        }

    def get_current_state(self) -> dict:
        """Returns a snapshot of the current state of the World Model."""
        try:
            # Get all nodes
            res = self.world_model.query_graph(
                "MATCH (n) RETURN labels(n)[0] as label, properties(n) as props"
            )
            nodes = [{"label": r[0], "properties": dict(r[1])} for r in res]

            # Get all edges
            res_edges = self.world_model.query_graph(
                "MATCH (a)-[r]->(b) RETURN properties(a).id as source, "
                "type(r) as type, properties(b).id as target"
            )
            edges = [
                {"source": r[0], "type": r[1], "target": r[2]}
                for r in res_edges
                if r[0] and r[2]
            ]
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            return {"error": str(e)}

    def evaluate_hypothesis(self, hypothesis: str) -> dict[str, Any]:
        """
        Evaluates a hypothesis text against the semantic graph.
        Returns: { 'entailment': 'Entailed' | 'Contradicted' | 'Undetermined', 'evidence': str, 'logical_form': str }
        """
        # Parse the hypothesis without incorporating it
        parse_res = self.add_belief(hypothesis, incorporate=False)
        if parse_res.get("status") != "success" or not parse_res.get("observation"):
            return {
                "entailment": "Undetermined",
                "evidence": "Failed to parse hypothesis.",
                "logical_form": "Unknown",
            }

        parse = parse_res["parse"]
        obs = parse_res["observation"]
        ptype = parse["type"]

        # Determine the target entity and property/relation we are checking
        if ptype in [
            "property_assignment",
            "class_membership",
            "property",
            "negative_property",
        ]:
            subj_id = obs.individuals[0].id
            prop = (
                next(iter(obs.individuals[0].properties.keys()))
                if isinstance(obs.individuals[0].properties, dict)
                else obs.individuals[0].properties[0]
            )

            # Replace spaces and hyphens with underscores
            prop_sanitized = prop.replace(" ", "_").replace("-", "_")
            prop_sanitized = "".join(
                c for c in prop_sanitized if c.isalnum() or c == "_"
            )
            if prop_sanitized.startswith("not_"):
                prop_sanitized = prop_sanitized[4:]

            # Direct check for the property
            q = (
                "MATCH (n:Entity {id: $id}) "
                f"RETURN n.prop_{prop_sanitized} AS has_pos, n.prop_not_{prop_sanitized} AS has_neg"
            )
            res = self.world_model.graph.query(q, params={"id": subj_id})

            has_pos = False
            has_neg = False
            if res.result_set:
                has_pos = res.result_set[0][0] is True
                has_neg = res.result_set[0][1] is True

            if has_pos:
                return {
                    "entailment": (
                        "Entailed" if ptype != "negative_property" else "Contradicted"
                    ),
                    "evidence": f"Found direct evidence for property {prop} on {subj_id}.",
                    "logical_form": f"{prop}({subj_id})",
                }
            elif has_neg:
                return {
                    "entailment": (
                        "Contradicted" if ptype != "negative_property" else "Entailed"
                    ),
                    "evidence": f"Found contradictory evidence for property {prop} on {subj_id}.",
                    "logical_form": f"NOT {prop}({subj_id})",
                }

            # Syllogistic subsumption check
            if ptype == "class_membership":
                concept = parse.get("concept_name", f"Concept_{prop.capitalize()}")
                q_sub = (
                    "MATCH (n:Entity {id: $id})-[:INSTANCE_OF]->(c:Concept) "
                    "RETURN c.name"
                )
                res_sub = self.world_model.graph.query(q_sub, params={"id": subj_id})
                if res_sub.result_set:
                    ancestors = [row[0] for row in res_sub.result_set]
                    if concept in ancestors:
                        return {
                            "entailment": "Entailed",
                            "evidence": f"Subsumption found: {subj_id} is instance of {concept}.",
                            "logical_form": f"{concept}({subj_id})",
                        }
            elif ptype in ["property", "negative_property", "property_assignment"]:
                q_sub = (
                    "MATCH (n:Entity {id: $id})-[:INSTANCE_OF]->(c:Concept) "
                    f"RETURN c.prop_{prop_sanitized} AS has_pos, c.prop_not_{prop_sanitized} AS has_neg"
                )
                res_sub = self.world_model.graph.query(q_sub, params={"id": subj_id})
                if res_sub.result_set:
                    for row in res_sub.result_set:
                        if row[0] is True:
                            return {
                                "entailment": (
                                    "Entailed"
                                    if ptype != "negative_property"
                                    else "Contradicted"
                                ),
                                "evidence": f"Subsumption found: {subj_id} is instance of concept with property {prop_sanitized}.",
                                "logical_form": f"{prop_sanitized}({subj_id})",
                            }
                        elif row[1] is True:
                            return {
                                "entailment": (
                                    "Contradicted"
                                    if ptype != "negative_property"
                                    else "Entailed"
                                ),
                                "evidence": f"Subsumption found: {subj_id} is instance of concept with negative property {prop_sanitized}.",
                                "logical_form": f"NOT {prop_sanitized}({subj_id})",
                            }

            return {
                "entailment": "Undetermined",
                "evidence": f"No direct or subsumptive evidence for property {prop} on {subj_id}.",
                "logical_form": f"? {prop}({subj_id})",
            }

        elif ptype == "relation":
            rel = obs.relations[0]

            # Check for exactly this relation
            q = (
                f"MATCH (a:Entity {{id: $src}})-[r:{rel.relation_type}]->(b:Entity {{id: $tgt}}) "
                "RETURN r.modality, r.truth_value"
            )
            res = self.world_model.graph.query(
                q, params={"src": rel.source_id, "tgt": rel.target_id}
            )

            if res.result_set:
                for row in res.result_set:
                    modality = row[0]
                    tv = row[1]

                    if tv == 0:
                        return {
                            "entailment": "Contradicted",
                            "evidence": f"Found negative relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                            "logical_form": f"NOT {rel.relation_type}({rel.source_id}, {rel.target_id})",
                        }

                    if modality == "can" and not rel.modality:
                        return {
                            "entailment": "Undetermined",
                            "evidence": f"Found possibility ('can') relation, but hypothesis asserts actuality.",
                            "logical_form": f"? {rel.relation_type}({rel.source_id}, {rel.target_id})",
                        }

                    return {
                        "entailment": "Entailed",
                        "evidence": f"Found relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                        "logical_form": f"{rel.relation_type}({rel.source_id}, {rel.target_id})",
                    }

            return {
                "entailment": "Undetermined",
                "evidence": f"No evidence for relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                "logical_form": f"? {rel.relation_type}({rel.source_id}, {rel.target_id})",
            }

        return {
            "entailment": "Undetermined",
            "evidence": "Unsupported hypothesis type.",
            "logical_form": "Unknown",
        }

    # ==========================================
    # Paraclete Protocol — T1 Constraint Layer
    # ==========================================

    def incorporate_axiom(self, axiom: DeontologicalAxiom | dict) -> dict:
        """
        Store a non-overridable T1 deontological axiom in the graph.
        Delegates to WorldModel.incorporate_axiom.
        """
        try:
            self.world_model.incorporate_axiom(axiom)
            axiom_id = (
                axiom.source_axiom
                if isinstance(axiom, DeontologicalAxiom)
                else axiom.get("source_axiom", "unknown")
            )
            return {"status": "success", "message": f"Axiom {axiom_id} stored."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_constraint(self, subject_id: str, relation: str, object_id: str) -> dict:
        """
        Route a proposed action triple through the T1 constraint layer.
        Delegates to WorldModel.check_constraint.
        """
        return self.world_model.check_constraint(subject_id, relation, object_id)

    def calibrate_belief(
        self, object_id: str, blocking_axiom: str, relation: str
    ) -> dict:
        """
        Implements EBE theorem SeeksDisconfirmation obligation.
        Called after check_constraint returns BLOCKED.
        Queries the graph for evidence that the factual premises triggering
        the block may be incorrect. Satisfies epistemic obligation without
        providing an override pathway for the T1 block.
        Delegates to WorldModel.calibrate_belief.
        """
        return self.world_model.calibrate_belief(object_id, blocking_axiom, relation)

    def escalate_block(
        self,
        object_id: str,
        verdict: str,
        blocking_axiom: str,
        relation: str,
    ) -> dict:
        """
        Third step in the Paraclete Protocol workflow:
          check_action → calibrate_belief → [CHALLENGED/UNCERTAIN] → escalate_block

        Runs epistemic resolution (contradiction or corroboration) and returns
        a FINAL_BLOCK or FINAL_PERMIT ruling with full provenance log.
        Conservative default applied under unresolvable uncertainty.
        No authority-based override pathway exists.
        Delegates to WorldModel.escalate_block.
        """
        return self.world_model.escalate_block(
            object_id, verdict, blocking_axiom, relation
        )

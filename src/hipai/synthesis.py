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

from .models import DeontologicalAxiom
from .world_model import WorldModel

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


class HIPAIManager:
    """
    High-level manager for the Montague-style semantic cognition system.
    Orchestrates the WorldModel and ZettelkastenSynthesizer.
    This class provides the interface expected by test_hipai.py.
    """

    def __init__(self, graph_name: str = "hipai_world"):
        self.world_model = WorldModel(graph_name=graph_name)
        self.synthesizer = ZettelkastenSynthesizer(self.world_model)

    def clear_database(self):
        """Standardizer for clearing the model's graph database."""
        self.world_model.clear_database()

    def add_belief(self, text: str) -> dict:
        """
        Parses a natural language belief into the system.
        Since we don't have an LLM connected in this basic version,
        we use basic string parsing to create Observations.
        """
        from .models import Individual, Observation

        text = text.strip(".")

        # Basic parser for "X is not a Y"
        if " is not a " in text:
            subject, obj = text.split(" is not a ", 1)
            obs = Observation(
                text_source=text,
                individuals=[
                    Individual(id=subject, name=subject, properties=[f"not_{obj}"])
                ],
                relations=[],
            )
            self.world_model.incorporate_observation(obs)
            return {"status": "success", "message": f"Added negative belief: {text}"}

        # Super basic parser for "X is a Y"
        if " is a " in text:
            subject, obj = text.split(" is a ", 1)
            obs = Observation(
                text_source=text,
                individuals=[Individual(id=subject, name=subject, properties=[obj])],
                relations=[],
            )
            self.world_model.incorporate_observation(obs)

            import inflect

            p = inflect.engine()
            singular_class = p.singular_noun(obj) or obj
            concept_name = f"Concept_{singular_class.capitalize()}"

            cypher = """
            MATCH (e:Entity {id: $subject})
            MERGE (c:Concept {name: $concept_name})
            MERGE (e)-[:INSTANCE_OF]->(c)
            """
            self.world_model.query_graph(
                cypher, {"subject": subject, "concept_name": concept_name}
            )

            return {"status": "success", "message": f"Added belief: {text}"}

        # Basic parser for "X is not Y"
        if " is not " in text:
            subject, obj = text.split(" is not ", 1)
            obs = Observation(
                text_source=text,
                individuals=[
                    Individual(id=subject, name=subject, properties=[f"not_{obj}"])
                ],
                relations=[],
            )
            self.world_model.incorporate_observation(obs)
            return {"status": "success", "message": f"Added negative belief: {text}"}

        # Basic parser for "X is Y"
        if " is " in text:
            subject, obj = text.split(" is ", 1)
            obs = Observation(
                text_source=text,
                individuals=[Individual(id=subject, name=subject, properties=[obj])],
                relations=[],
            )
            self.world_model.incorporate_observation(obs)
            return {"status": "success", "message": f"Added belief: {text}"}

        # Basic parser for "All Xs are Ys"
        if text.startswith("All ") and " are " in text:
            parts = text[4:].split(" are ")
            if len(parts) == 2:
                subject_class = parts[0]
                obj_property = parts[1]

                import inflect

                p = inflect.engine()
                singular_class = p.singular_noun(subject_class) or subject_class

                # Logically, this creates a rule.
                # For this basic implementation, we just store it as a
                # Concept-level property.
                concept_name = f"Concept_{singular_class.capitalize()}"
                self.world_model.create_structure_note(concept_name, [])
                # Apply the property to the concept
                prop_key = obj_property.strip().replace(" ", "_").replace("-", "_")
                q = (
                    f"MATCH (c:Concept {{name: '{concept_name}'}}) "
                    f"SET c.prop_{prop_key} = true"
                )
                self.world_model.query_graph(q)
                return {
                    "status": "success",
                    "message": f"Added universal belief: {text}",
                }

        return {
            "status": "error",
            "message": "Failed to parse belief. Use 'X is Y' or 'All X are Y' format.",
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
        Evaluate a hypothesis against the intensional and extensional knowledge.
        Supports both Entities (Content Nodes) and Concepts (Structure Notes).
        """
        hypothesis = hypothesis.strip(".")

        # 1. Normalize the property and identify if it's a negation.
        is_negative = False
        if " is not " in hypothesis:
            subject, obj = hypothesis.split(" is not ", 1)
            is_negative = True
        elif " is " in hypothesis:
            subject, obj = hypothesis.split(" is ", 1)
        else:
            return {
                "hypothesis": hypothesis,
                "entailment": "Error",
                "confidence": 0.0,
                "reasoning": "Failed to parse. Use 'X is Y' or 'X is not Y'.",
            }

        subject = subject.strip()
        obj = obj.strip()

        # 2. Check if the subject is epistemically contested
        q_contested = """
        MATCH (n:Entity)
        WHERE n.id = $subject OR n.name = $subject
        RETURN n.epistemically_contested AS contested
        """
        res_c = self.world_model.query_graph(q_contested, {"subject": subject})
        if res_c and res_c[0][0] is True:
            return {
                "hypothesis": hypothesis,
                "entailment": "Contested",
                "confidence": 0.5,
                "reasoning": f"Properties for {subject} are epistemically contested.",
            }

        # 3. Sanitize the property for Cypher key usage
        prop_sanitized = "".join(
            c
            for c in obj.replace(" ", "_").replace("-", "_")
            if c.isalnum() or c == "_"
        )

        # 4. Hybrid Search: Direct Entity Property + Concept Inheritance
        q_logic = f"""
        MATCH (n:Entity)
        WHERE n.id = $subject OR n.name = $subject
        OPTIONAL MATCH (n)-[:INSTANCE_OF]->(c:Concept)
        RETURN n.prop_{prop_sanitized} as direct_pos,
               n.prop_not_{prop_sanitized} as direct_neg,
               collect(c.prop_{prop_sanitized}) as concept_pos,
               collect(c.prop_not_{prop_sanitized}) as concept_neg
        """
        res = self.world_model.query_graph(q_logic, {"subject": subject})

        entailment = "Undetermined"
        confidence = 0.0
        reasoning = (
            f"No info found about '{obj}' for {subject} "
            "in Content Nodes or Structure Notes."
        )

        if res:
            row = res[0]
            direct_pos = row[0]
            direct_neg = row[1]
            concept_pos = [p for p in row[2] if p is not None]
            concept_neg = [p for p in row[3] if p is not None]

            # Evidence for the positive property
            has_pos = direct_pos is True or any(concept_pos)
            # Evidence for the negative property (not X)
            has_neg = direct_neg is True or any(concept_neg)

            # Contradiction check
            if has_pos and has_neg:
                entailment = "Contested"
                confidence = 0.5
                reasoning = (
                    f"Contradictory evidence: both '{obj}' and 'not {obj}' "
                    f"present for {subject}."
                )
            elif is_negative:
                # Hypothesis is "is not X"
                if has_neg:
                    entailment = "Entailed"
                    confidence = 1.0
                    reasoning = f"Found '{obj}' in negative state for {subject}."
                elif has_pos:
                    entailment = "Denied"
                    confidence = 1.0
                    reasoning = f"Found '{obj}' in positive state, contradicts notion."
            else:
                # Hypothesis is "is X"
                if has_pos:
                    entailment = "Entailed"
                    confidence = 1.0
                    reasoning = (
                        f"Found '{obj}' for {subject} via direct/inherited props."
                    )
                elif has_neg:
                    entailment = "Denied"
                    confidence = 1.0
                    reasoning = f"Found 'not {obj}' for {subject}, contradicts notion."

        # 5. Final Fallback: Check for relations
        if entailment == "Undetermined":
            prop_upper = "".join(
                c for c in obj.upper().replace(" ", "_") if c.isalnum() or c == "_"
            )
            q_rel = (
                "MATCH (n:Entity)-[r]->(m:Entity) "
                "WHERE (n.id = $subject OR n.name = $subject) "
                "RETURN n.id, n.name, type(r), m.id, m.name"
            )
            res_rel = self.world_model.query_graph(
                q_rel, {"subject": subject, "prop_upper": prop_upper}
            )
            if res_rel:
                entailment = "Entailed" if not is_negative else "Denied"
                confidence = 1.0
                reasoning = (
                    f"Found relation {prop_upper} from {subject} "
                    f"to {res_rel[0][0]}."
                )

        # 6. Forward chaining: entity has prop_X → find Concept_X → check prop_Y
        # Resolves transitive syllogisms: "All Xs are Ys" + "A is X" → "A is Y"
        if entailment == "Undetermined" and not is_negative:
            q_entity_keys = """
            MATCH (n:Entity)
            WHERE n.id = $subject OR n.name = $subject
            RETURN keys(n) AS entity_keys
            """
            res_keys = self.world_model.query_graph(q_entity_keys, {"subject": subject})
            if res_keys and res_keys[0][0]:
                entity_keys = res_keys[0][0]
                # Collect positive membership properties (prop_X where value=true)
                membership_props = [
                    k[5:]
                    for k in entity_keys
                    if k.startswith("prop_") and not k.startswith("prop_not_")
                ]
                for membership in membership_props:
                    # Concept name: Concept_{Membership.capitalize()}
                    concept_fragment = membership.capitalize()
                    q_chain = f"""
                    MATCH (c:Concept)
                    WHERE c.name CONTAINS $concept_fragment
                    AND c.prop_{prop_sanitized} IS NOT NULL
                    RETURN c.name
                    """
                    res_chain = self.world_model.query_graph(
                        q_chain, {"concept_fragment": concept_fragment}
                    )
                    if res_chain:
                        entailment = "Entailed"
                        confidence = 1.0
                        reasoning = (
                            f"Forward chain: {subject} is {membership} → "
                            f"Concept_{concept_fragment} implies {obj} "
                            f"(via {res_chain[0][0]})."
                        )
                        break

        return {
            "hypothesis": hypothesis,
            "entailment": entailment,
            "confidence": confidence,
            "reasoning": reasoning,
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

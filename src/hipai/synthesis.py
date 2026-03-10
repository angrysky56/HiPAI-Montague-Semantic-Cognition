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
from typing import List, Dict, Optional
try:
    from sklearn.cluster import KMeans
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

from .world_model import WorldModel


class ZettelkastenSynthesizer:
    """
    Synthesizes higher-order logic structures (Structure Notes and Main Structure Notes)
    from base observations (Content Nodes).
    Implements intensional logic evaluation by allowing worlds (graph states) to be
    evaluated over time.
    """
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def synthesize_concepts(self, property_threshold: int = 1) -> List[str]:
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
        except Exception:
            return [] # In case the graph doesn't exist

        all_props = set()
        for row in result:
            keys = row[0]
            for key in keys:
                if key.startswith('prop_'):
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

    def vector_synthesize_concepts(self, n_clusters: int = 2) -> List[str]:
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

    def synthesize_domains(self, concept_threshold: int = 2) -> List[str]:
        """
        Generates Main Structure Notes (Domains) by clustering related Concepts.
        For simplicity, this clusters concepts that share underlying entities.
        """
        # Find concepts that share at least one entity
        q = """
        MATCH (c1:Concept)<-[:INSTANCE_OF]-(e:Entity)-[:INSTANCE_OF]->(c2:Concept)
        WHERE id(c1) < id(c2)
        RETURN c1.name, c2.name
        """
        try:
            res = self.world_model.query_graph(q)
        except Exception:
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

    def add_belief(self, text: str) -> Dict:
        """
        Parses a natural language belief into the system.
        Since we don't have an LLM connected in this basic version,
        we use basic string parsing to create Observations.
        """
        from .models import Individual, Observation
        
        text = text.strip(".")
        
        # Super basic parser for "X is a Y"
        if " is a " in text:
            subject, obj = text.split(" is a ", 1)
            obs = Observation(
                text_source=text,
                individuals=[Individual(id=subject, name=subject, properties=[obj])],
                relations=[]
            )
            self.world_model.incorporate_observation(obs)
            return {"status": "success", "message": f"Added belief: {text}"}
            
        # Basic parser for "X is Y"
        if " is " in text:
            subject, obj = text.split(" is ", 1)
            obs = Observation(
                text_source=text,
                individuals=[Individual(id=subject, name=subject, properties=[obj])],
                relations=[]
            )
            self.world_model.incorporate_observation(obs)
            return {"status": "success", "message": f"Added belief: {text}"}
            
        # Basic parser for "All Xs are Ys"
        if text.startswith("All ") and " are " in text:
            parts = text[4:].split(" are ")
            if len(parts) == 2:
                subject_class = parts[0]
                obj_property = parts[1]
                # Logically, this creates a rule. 
                # For this basic implementation, we just store it as a Concept-level property.
                self.world_model.create_structure_note(f"Concept_{subject_class.capitalize()}", [])
                # Apply the property to the concept
                q = f"MATCH (c:Concept {{name: 'Concept_{subject_class.capitalize()}'}}) SET c.prop_{obj_property} = true"
                self.world_model.query_graph(q)
                return {"status": "success", "message": f"Added universal belief: {text}"}

        return {"status": "error", "message": "Failed to parse belief. Use 'X is Y' or 'All X are Y' format."}

    def get_current_state(self) -> Dict:
        """Returns a snapshot of the current state of the World Model."""
        try:
            # Get all nodes
            res = self.world_model.query_graph("MATCH (n) RETURN labels(n)[0] as label, properties(n) as props")
            nodes = [{"label": r[0], "properties": dict(r[1])} for r in res]
            
            # Get all edges
            res_edges = self.world_model.query_graph("MATCH (a)-[r]->(b) RETURN properties(a).id as source, type(r) as type, properties(b).id as target")
            edges = [{"source": r[0], "type": r[1], "target": r[2]} for r in res_edges if r[0] and r[2]]
            
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            return {"error": str(e)}

    def evaluate_hypothesis(self, hypothesis: str) -> Dict:
        """
        Evaluates a hypothesis such as 'Socrates is mortal'.
        """
        hypothesis = hypothesis.strip(".")
        if " is " in hypothesis:
            subject, obj = hypothesis.split(" is ", 1)
            
            # Check if this property exists directly on the entity
            q_direct = f"MATCH (n:Entity {{id: $subject}}) WHERE n.prop_{obj} = true RETURN n"
            try:
                res = self.world_model.query_graph(q_direct, {"subject": subject})
                if len(res) > 0:
                    return {
                        "hypothesis": hypothesis,
                        "entailment": "True",
                        "confidence": 1.0,
                        "reasoning": f"Found direct evidence that {subject} has property {obj}."
                    }
            except Exception:
                pass
                
            # Check if entity belongs to a Concept that has this property (syllogism)
            # Find classes Socrates belongs to
            q_concept = f"""
            MATCH (e:Entity {{id: $subject}})-[:INSTANCE_OF]->(c:Concept)
            WHERE c.prop_{obj} = true
            RETURN c.name
            """
            try:
                res2 = self.world_model.query_graph(q_concept, {"subject": subject})
                if len(res2) > 0:
                    concept = res2[0][0]
                    return {
                        "hypothesis": hypothesis,
                        "entailment": "True",
                        "confidence": 1.0,
                        "reasoning": f"{subject} is an instance of {concept}, which has property {obj}."
                    }
            except Exception:
                pass

            # Maybe Zettelkasten Synthesis needs to run?
            # E.g., if we know Socrates is a man, and we know All men are mortal,
            # we can synthesize a Concept_man if we parse it right.
            # In our simple parser, "man" is a property.
            
            # Check if entity has some property 'prop_X' 
            # and there exists a Concept_X with property 'prop_Y'
            try:
                # Find properties of the entity
                q_props = "MATCH (e:Entity {id: $subject}) RETURN keys(e)"
                res_props = self.world_model.query_graph(q_props, {"subject": subject})
                if res_props:
                    for key in res_props[0][0]:
                        if key.startswith("prop_"):
                            prop_name = key[5:]
                            # Try matching Concept with capitalized name and variations
                            concept_variants = [
                                f"Concept_{prop_name.capitalize()}",
                                f"Concept_{prop_name}",
                                f"Concept_{prop_name}s", # Simple pluralization
                                f"Concept_{prop_name}es", # Simple pluralization
                                f"Concept_{prop_name[:-2].capitalize()}en" if prop_name.lower().endswith('man') else None # man -> men
                            ]
                            for cv in filter(None, concept_variants):
                                q_inf = f"MATCH (c:Concept {{name: '{cv}'}}) WHERE c.prop_{obj} = true RETURN c.name"
                                res_inf = self.world_model.query_graph(q_inf)
                                if res_inf:
                                    return {
                                        "hypothesis": hypothesis,
                                        "entailment": "True",
                                        "confidence": 1.0,
                                        "reasoning": f"{subject} is a {prop_name}, and all {cv.replace('Concept_', '')} are {obj}."
                                    }
            except (ValueError, KeyError, TypeError) as e:
                print(f"Error extracting knowledge: {e}")
                pass
                
        return {
            "hypothesis": hypothesis,
            "entailment": "Unknown", 
            "confidence": 0.0,
            "reasoning": "Could not find evidence in the knowledge graph to prove or disprove."
        }

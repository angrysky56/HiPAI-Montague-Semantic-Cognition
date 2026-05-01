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

    # Map of verb forms to their canonical base/stem for relation type matching.
    # If a verb matches as a regex key, use the corresponding base form.
    _VERB_STEM_OVERRIDES: dict[str, str] = {
        "causes": "cause", "leads": "lead", "produces": "produce",
        "creates": "create", "triggers": "trigger", "generates": "generate",
        "enables": "enable", "prevents": "prevent", "blocks": "block",
        "inhibits": "inhibit", "harms": "harm",
        "exploits": "exploit", "manipulates": "manipulate",
        "influences": "influence", "affects": "affect", "impacts": "impact",
        "shapes": "shape", "alters": "alter", "modifies": "modify",
        "requires": "require", "needs": "need",
        "supports": "support", "confirms": "confirm",
        "contradicts": "contradict", "challenges": "challenge",
        "undermines": "undermine",
        "visits": "visit", "sees": "see", "meets": "meet",
        "calls": "call", "loves": "love", "hates": "hate",
    }

    @staticmethod
    def _normalize_verb(verb: str) -> str:
        """Normalize an inflected verb to its base/stem form for relation type creation.

        Uses an explicit override table for known verbs, then falls back to
        simple suffix stripping (``-es`` → ``-e``, ``-s`` → base).
        """
        v = verb.lower().strip()
        if v in HIPAIManager._VERB_STEM_OVERRIDES:
            return HIPAIManager._VERB_STEM_OVERRIDES[v]
        # Fallback heuristics
        if v.endswith("ies"):          # e.g. "relies" → "rely"
            return v[:-3] + "y"
        if v.endswith("ses") or v.endswith("zes") or v.endswith("xes") or v.endswith("ches") or v.endswith("shes"):
            return v[:-2]              # e.g. "causes" already handled above
        if v.endswith("es"):
            return v[:-1]              # e.g. "produces" → "produce"
        if v.endswith("s") and not v.endswith("ss"):
            return v[:-1]              # e.g. "harms" → "harm"
        return v

    def __init__(self, graph_name: str = "hipai_world"):
        self.world_model = WorldModel(graph_name=graph_name)
        self.synthesizer = ZettelkastenSynthesizer(self.world_model)
        self.logger = logger

    def clear_database(self):
        """Standardizer for clearing the model's graph database."""
        self.world_model.clear_database()

    def add_belief(self, text: str, incorporate: bool = True) -> dict[str, Any]:
        """
        Synthesize a belief from natural language text and add it to the graph if incorporate is True.
        Handles contradictions and logic routing.
        
        If multiple interpretations are found, raises AmbiguityDetectedError.
        If no structured patterns match, falls back to unstructured belief.
        """
        from .models import Individual, Observation, Relation
        from .exceptions import AmbiguityDetectedError
        import re
        import inflect

        text = text.strip().strip(".")

        # ─── Pattern 11: Attitude Verbs (Checked first to avoid ambiguity) ───
        attitude_match = re.match(r"^(.+?)\s+(believes?|knows?|says?|thinks?)\s+that\s+(.+)$", text, re.IGNORECASE)
        if attitude_match:
            subject = attitude_match.group(1).strip()
            verb = attitude_match.group(2).strip().lower()
            proposition = attitude_match.group(3).strip()
            
            # Factive attitudes (e.g. knows) entail their propositions.
            # Non-factive attitudes do not.
            is_factive = verb in ["knows", "know"]
            
            # Recursively call self.add_belief to get the nested Observation
            # Only incorporate if factive, to avoid polluting the graph with non-factive properties
            res = self.add_belief(proposition, incorporate=is_factive)
            if res.get("status") == "success" and "observation" in res:
                nested_obs = res["observation"]
                
                # If it's non-factive, we STILL need to create the Observation node itself 
                # to point the attitude relation to it, but without asserting its properties.
                if not is_factive:
                    self.world_model.graph.query(
                        "MERGE (o:EpistemicNode:Observation {event_id: $event_id}) "
                        "SET o.text_source = $text_source, o.modality = $modality",
                        params={
                            "event_id": nested_obs.event_id,
                            "text_source": nested_obs.text_source,
                            "modality": nested_obs.modality
                        }
                    )
                
                attitude_rel_type = verb.upper()
                
                obs = Observation(
                    text_source=text,
                    individuals=[Individual(id=subject, name=subject)],
                    relations=[],
                    tense="present",
                )
                
                # Link subject -> nested_observation
                if incorporate:
                    self.world_model.graph.query(
                        f"""
                        MERGE (o:EpistemicNode:Observation {{event_id: $event_id}})
                        MERGE (e:ContentNode:Entity {{id: $subject}})
                        MERGE (nested:EpistemicNode:Observation {{event_id: $nested_event_id}})
                        MERGE (e)-[r:{attitude_rel_type}]->(nested)
                        SET r.is_factive = $is_factive
                        MERGE (o)-[:OBSERVED]->(e)
                        """,
                        params={
                            "event_id": obs.event_id,
                            "subject": subject,
                            "nested_event_id": nested_obs.event_id,
                            "is_factive": is_factive
                        }
                    )
                
                parse = {
                    "observation": obs,
                    "type": "attitude_belief",
                    "attitude": verb,
                    "subject": subject,
                    "nested_observation": nested_obs
                }
                
                return {
                    "status": "success", 
                    "message": f"Successfully parsed and {'added' if incorporate else 'processed'} attitude belief.",
                    "observation": obs,
                    "parse": parse
                }
            else:
                return res # return nested error

        possible_parses = []

        # ─── Pattern 1: "X is not a Y" / "X was not a Y" / "X will not be a Y" ───
        neg_a_seps = [
            (" will not be a ", "future", None), (" will not be an ", "future", None),
            (" was not a ", "past", None), (" was not an ", "past", None), (" were not a ", "past", None), (" were not an ", "past", None),
            (" is not a ", "present", None), (" is not an ", "present", None), (" are not a ", "present", None), (" are not an ", "present", None),
            (" must not be a ", "present", "must"), (" must not be an ", "present", "must"),
            (" cannot be a ", "present", "can"), (" cannot be an ", "present", "can"), (" can not be a ", "present", "can"), (" can not be an ", "present", "can"),
            (" should not be a ", "present", "should"), (" should not be an ", "present", "should")
        ]
        for sep, t, mod in neg_a_seps:
            if sep in text and not text.startswith(("All ", "Some ", "No ")):
                subject, obj = text.split(sep, 1)
                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=subject.strip(), name=subject.strip(),
                                   properties=[f"not_{obj.strip()}"])
                    ],
                    relations=[],
                    tense=t,
                    modality=mod
                )
                possible_parses.append({"observation": obs, "type": "negative_property_a", "tense": t})
                break

        # ─── Pattern 2: "X is a Y" / "X was a Y" / "X will be a Y" ───
        pos_a_seps = [
            (" will be a ", "future", None), (" will be an ", "future", None),
            (" was a ", "past", None), (" was an ", "past", None), (" were a ", "past", None), (" were an ", "past", None),
            (" is a ", "present", None), (" is an ", "present", None), (" are a ", "present", None), (" are an ", "present", None),
            (" must be a ", "present", "must"), (" must be an ", "present", "must"),
            (" can be a ", "present", "can"), (" can be an ", "present", "can"),
            (" should be a ", "present", "should"), (" should be an ", "present", "should")
        ]
        for sep, t, mod in pos_a_seps:
            if sep in text and not text.startswith(("All ", "Some ", "No ")):
                subject, obj = text.split(sep, 1)
                subject = subject.strip()
                obj = obj.strip()
                obs = Observation(
                    text_source=text,
                    individuals=[Individual(id=subject, name=subject, properties=[obj])],
                    relations=[],
                    tense=t,
                    modality=mod
                )
                
                p = inflect.engine()
                singular_class = p.singular_noun(obj) or obj
                concept_name = f"Concept_{singular_class.capitalize()}"
                possible_parses.append({
                    "observation": obs, 
                    "type": "class_membership",
                    "concept_name": concept_name,
                    "tense": t
                })
                break

        # ─── Pattern 3: "All X are Y" ───
        if text.startswith("All ") and " are " in text:
            parts = text[4:].split(" are ")
            if len(parts) == 2:
                subject_class = parts[0].strip()
                obj_property = parts[1].strip()
                p = inflect.engine()
                singular_class = p.singular_noun(subject_class) or subject_class
                concept_name = f"Concept_{singular_class.capitalize()}"
                prop_key = obj_property.replace(" ", "_").replace("-", "_")
                prop_key = "".join(c for c in prop_key if c.isalnum() or c == "_")
                
                # For universal beliefs, we represent them slightly differently in the model
                # (usually directly on the Concept node in the graph)
                possible_parses.append({
                    "observation": None, # Universal beliefs don't map to a single Entity observation easily
                    "type": "universal_belief",
                    "concept_name": concept_name,
                    "property_key": f"prop_{prop_key}"
                })

        # ─── Pattern 4: "X is not Y" / "X was not Y" / "X will not be Y" ───
        neg_seps = [
            (" will not be ", "future", None), (" was not ", "past", None), (" were not ", "past", None), (" is not ", "present", None), (" are not ", "present", None),
            (" must not be ", "present", "must"), (" cannot be ", "present", "can"), (" can not be ", "present", "can"), (" should not be ", "present", "should")
        ]
        for sep, t, mod in neg_seps:
            if sep in text and f"{sep}a " not in text and f"{sep}an " not in text and not text.startswith(("All ", "Some ", "No ")):
                subject, obj = text.split(sep, 1)
                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=subject.strip(), name=subject.strip(),
                                   properties=[f"not_{obj.strip()}"])
                    ],
                    relations=[],
                    tense=t,
                    modality=mod
                )
                possible_parses.append({"observation": obs, "type": "negative_property", "tense": t})
                break

        # ─── Pattern 5: "X is Y" / "X was Y" / "X will be Y" ───
        pos_seps = [
            (" will be ", "future", None), (" was ", "past", None), (" were ", "past", None), (" is ", "present", None), (" are ", "present", None),
            (" must be ", "present", "must"), (" can be ", "present", "can"), (" should be ", "present", "should")
        ]
        for sep, t, mod in pos_seps:
            # Check to avoid overlapping with "is a", "is not", "All X are Y", etc.
            if sep in text and not text.startswith(("All ", "Some ", "No ")) and not any(s[0] in text for s in neg_a_seps + pos_a_seps + neg_seps):
                subject, obj = text.split(sep, 1)
                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=subject.strip(), name=subject.strip(),
                                   properties=[obj.strip()])
                    ],
                    relations=[],
                    tense=t,
                    modality=mod
                )
                possible_parses.append({"observation": obs, "type": "property_assignment", "tense": t})
                break

        # ─── Pattern 6: "X has/have Y" ───
        for sep in (" has ", " have "):
            if sep in text:
                subject, obj = text.split(sep, 1)
                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=subject.strip(), name=subject.strip(),
                                   properties=[obj.strip()])
                    ],
                    relations=[],
                )
                possible_parses.append({"observation": obs, "type": "possession"})

        # ─── Pattern 7: "X are Y" (without "All/Some/No") ───
        if " are " in text and not text.startswith(("All ", "Some ", "No ")):
            subject, obj = text.split(" are ", 1)
            obs = Observation(
                text_source=text,
                individuals=[
                    Individual(id=subject.strip(), name=subject.strip(),
                               properties=[obj.strip()])
                ],
                relations=[],
            )
            possible_parses.append({"observation": obs, "type": "plural_property"})

        # ─── Pattern 8: Relational verbs → create a relation ───
        relational_patterns = [
            (r"^(.+?)\s+(?:(will|did|had|was|were|has|have|must|can|should)\s+)?(causes?|leads?\s+to|produces?|creates?|triggers?|generates?|enables?|prevents?|blocks?|inhibits?|harms?)\s+(.+)$", "causal"),
            (r"^(.+?)\s+(?:(will|did|had|was|were|has|have|must|can|should)\s+)?(exploits?|manipulates?|influences?|affects?|impacts?|shapes?|alters?|modifies?)\s+(.+)$", "influence"),
            (r"^(.+?)\s+(?:(will|did|had|was|were|has|have|must|can|should)\s+)?(requires?|needs?|depends?\s+on|relies?\s+on)\s+(.+)$", "dependency"),
            (r"^(.+?)\s+(?:(will|did|had|was|were|has|have|must|can|should)\s+)?(supports?|confirms?|contradicts?|challenges?|undermines?)\s+(.+)$", "epistemic"),
            (r"^(.+?)\s+(?:(will|did|had|was|were|has|have|must|can|should)\s+)?(visits?|sees?|meets?|calls?|loves?|hates?)\s+(.+)$", "social"),
        ]
        for pattern, rel_category in relational_patterns:
            m = re.match(pattern, text, re.IGNORECASE)
            if m and not text.startswith(("All ", "Some ", "No ")):
                subject = m.group(1).strip()
                aux = m.group(2)
                verb = m.group(3).strip()
                obj = m.group(4).strip()
                # Normalize verb to stem form so 'harms' → 'HARM' matches axioms
                verb_stem = self._normalize_verb(verb)
                rel_type = verb_stem.upper().replace(" ", "_")
                rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")

                # Tense and Modality detection for relations
                tense = "present"
                modality = None
                if aux:
                    aux_lower = aux.lower()
                    if aux_lower in ["must", "can", "should"]:
                        modality = aux_lower
                        
                if re.search(r"\bwill\s+", text, re.IGNORECASE):
                    tense = "future"
                elif re.search(r"\b(did|had|was|were)\s+", text, re.IGNORECASE) or verb.endswith("ed"):
                    tense = "past"

                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=subject, name=subject),
                        Individual(id=obj, name=obj),
                    ],
                    relations=[
                        Relation(source_id=subject, target_id=obj, relation_type=rel_type, tense=tense, modality=modality)
                    ],
                    tense=tense,
                    modality=modality
                )
                possible_parses.append({
                    "observation": obs, 
                    "type": "relation", 
                    "category": rel_category,
                    "rel_type": rel_type
                })

        # ─── Pattern 9: "No X are/is Y" (Negative Universal) ───
        if text.startswith("No ") and (" are " in text or " is " in text):
            sep = " are " if " are " in text else " is "
            parts = text[3:].split(sep)
            if len(parts) == 2:
                subject_class = parts[0].strip()
                obj_property = parts[1].strip()
                if obj_property.lower().startswith("a "): obj_property = obj_property[2:].strip()
                elif obj_property.lower().startswith("an "): obj_property = obj_property[3:].strip()
                
                p = inflect.engine()
                singular_class = p.singular_noun(subject_class) or subject_class
                concept_name = f"Concept_{singular_class.capitalize()}"
                prop_key = obj_property.replace(" ", "_").replace("-", "_")
                prop_key = "".join(c for c in prop_key if c.isalnum() or c == "_")
                
                possible_parses.append({
                    "observation": None,
                    "type": "negative_universal_belief",
                    "concept_name": concept_name,
                    "property_key": f"prop_not_{prop_key}"
                })

        # ─── Pattern 10: "Some X are/is Y" (Existential) ───
        if text.startswith("Some ") and (" are " in text or " is " in text):
            sep = " are " if " are " in text else " is "
            parts = text[5:].split(sep)
            if len(parts) == 2:
                subject_class = parts[0].strip()
                obj_property = parts[1].strip()
                if obj_property.lower().startswith("a "): obj_property = obj_property[2:].strip()
                elif obj_property.lower().startswith("an "): obj_property = obj_property[3:].strip()
                
                p = inflect.engine()
                singular_class = p.singular_noun(subject_class) or subject_class
                concept_name = f"Concept_{singular_class.capitalize()}"
                
                from uuid import uuid4
                anon_id = f"anonymous_{uuid4().hex[:8]}"
                
                obs = Observation(
                    text_source=text,
                    individuals=[
                        Individual(id=anon_id, name=anon_id, properties=[obj_property])
                    ],
                    relations=[]
                )
                
                possible_parses.append({
                    "observation": obs,
                    "type": "existential_belief",
                    "concept_name": concept_name,
                    "subject_id": anon_id
                })



        # ─── Pruning ───
        valid_parses = []
        for parse in possible_parses:
            if parse["type"] in ["relation", "causal", "influence", "dependency", "epistemic", "social"]:
                obs = parse["observation"]
                is_valid = True
                for rel in obs.relations:
                    # Check if this relation is forbidden by T1 constraints
                    res = self.world_model.check_constraint(rel.source_id, rel.relation_type, rel.target_id)
                    if not res["permitted"]:
                        is_valid = False
                        break
                if is_valid:
                    valid_parses.append(parse)
            else:
                # For now, we assume property assignments, universal beliefs, and attitudes are valid.
                # Contradiction detection for properties happens during incorporation.
                valid_parses.append(parse)

        # ─── Ambiguity Check ───
        if len(valid_parses) > 1:
            raise AmbiguityDetectedError(valid_parses)

        # Fallback to LLM extraction if no simple pattern matches
        if not possible_parses:
            if incorporate:
                self.logger.warning(f"No patterns matched '{text}', falling back to LLM.")
                # Implement LLM extraction later
            return {"status": "error", "message": "Failed to parse text."}

        # If we have exactly one valid parse, commit it
        if len(valid_parses) == 1:
            parse = valid_parses[0]
            if incorporate and parse["observation"]:
                self.world_model.incorporate_observation(parse["observation"])
            
            if incorporate:
                # Handle special metadata logic (e.g., universal beliefs or concepts)
                if parse["type"] == "class_membership":
                    cypher = """
                    MATCH (e:Entity {id: $subject})
                    MERGE (c:Concept {name: $concept_name})
                    MERGE (e)-[:INSTANCE_OF]->(c)
                    """
                    self.world_model.query_graph(
                        cypher, {
                            "subject": parse["observation"].individuals[0].id, 
                            "concept_name": parse["concept_name"]
                        }
                    )
                elif parse["type"] == "universal_belief":
                    self.world_model.create_structure_note(parse["concept_name"], [])
                    q = (
                        f"MATCH (c:Concept {{name: '{parse['concept_name']}'}}) "
                        f"SET c.{parse['property_key']} = true"
                    )
                    self.world_model.query_graph(q)
                elif parse["type"] == "negative_universal_belief":
                    self.world_model.create_structure_note(parse["concept_name"], [])
                    q = (
                        f"MATCH (c:Concept {{name: '{parse['concept_name']}'}}) "
                        f"SET c.{parse['property_key']} = true"
                    )
                    self.world_model.query_graph(q)
                elif parse["type"] == "existential_belief":
                    # Create anonymous individual and link to concept
                    self.world_model.incorporate_observation(parse["observation"])
                    cypher = """
                    MATCH (e:Entity {id: $subject})
                    MERGE (c:Concept {name: $concept_name})
                    MERGE (e)-[:INSTANCE_OF]->(c)
                    """
                    self.world_model.query_graph(
                        cypher, {
                            "subject": parse["subject_id"], 
                            "concept_name": parse["concept_name"]
                        }
                    )

            return {
                "status": "success", 
                "message": f"Added belief: {text}",
                "observation": parse["observation"],
                "parse": parse
            }

        # ─── Fallback: store as free-text entity ───

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
            "message": f"Added as unstructured belief (no pattern matched): {text}",
            "observation": obs
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
                "logical_form": "Unknown"
            }
            
        parse = parse_res["parse"]
        obs = parse["observation"]
        ptype = parse["type"]
        
        # Determine the target entity and property/relation we are checking
        if ptype in ["property_assignment", "class_membership", "property", "negative_property"]:
            subj_id = obs.individuals[0].id
            prop = list(obs.individuals[0].properties.keys())[0] if isinstance(obs.individuals[0].properties, dict) else obs.individuals[0].properties[0]
            
            # Replace spaces and hyphens with underscores
            prop_sanitized = prop.replace(" ", "_").replace("-", "_")
            prop_sanitized = "".join(c for c in prop_sanitized if c.isalnum() or c == "_")
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
                    "entailment": "Entailed" if ptype != "negative_property" else "Contradicted",
                    "evidence": f"Found direct evidence for property {prop} on {subj_id}.",
                    "logical_form": f"{prop}({subj_id})"
                }
            elif has_neg:
                return {
                    "entailment": "Contradicted" if ptype != "negative_property" else "Entailed",
                    "evidence": f"Found contradictory evidence for property {prop} on {subj_id}.",
                    "logical_form": f"NOT {prop}({subj_id})"
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
                            "logical_form": f"{concept}({subj_id})"
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
                                "entailment": "Entailed" if ptype != "negative_property" else "Contradicted",
                                "evidence": f"Subsumption found: {subj_id} is instance of concept with property {prop_sanitized}.",
                                "logical_form": f"{prop_sanitized}({subj_id})"
                            }
                        elif row[1] is True:
                            return {
                                "entailment": "Contradicted" if ptype != "negative_property" else "Entailed",
                                "evidence": f"Subsumption found: {subj_id} is instance of concept with negative property {prop_sanitized}.",
                                "logical_form": f"NOT {prop_sanitized}({subj_id})"
                            }
            
            return {
                "entailment": "Undetermined",
                "evidence": f"No direct or subsumptive evidence for property {prop} on {subj_id}.",
                "logical_form": f"? {prop}({subj_id})"
            }
            
        elif ptype == "relation":
            rel = obs.relations[0]
            
            # Check for exactly this relation
            q = (
                f"MATCH (a:Entity {{id: $src}})-[r:{rel.relation_type}]->(b:Entity {{id: $tgt}}) "
                "RETURN r.modality, r.truth_value"
            )
            res = self.world_model.graph.query(q, params={"src": rel.source_id, "tgt": rel.target_id})
            
            if res.result_set:
                for row in res.result_set:
                    modality = row[0]
                    tv = row[1]
                    
                    if tv == 0:
                        return {
                            "entailment": "Contradicted",
                            "evidence": f"Found negative relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                            "logical_form": f"NOT {rel.relation_type}({rel.source_id}, {rel.target_id})"
                        }
                    
                    if modality == "can" and not rel.modality:
                        return {
                            "entailment": "Undetermined",
                            "evidence": f"Found possibility ('can') relation, but hypothesis asserts actuality.",
                            "logical_form": f"? {rel.relation_type}({rel.source_id}, {rel.target_id})"
                        }
                        
                    return {
                        "entailment": "Entailed",
                        "evidence": f"Found relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                        "logical_form": f"{rel.relation_type}({rel.source_id}, {rel.target_id})"
                    }
                    
            return {
                "entailment": "Undetermined",
                "evidence": f"No evidence for relation {rel.relation_type} between {rel.source_id} and {rel.target_id}.",
                "logical_form": f"? {rel.relation_type}({rel.source_id}, {rel.target_id})"
            }

        return {
            "entailment": "Undetermined",
            "evidence": "Unsupported hypothesis type.",
            "logical_form": "Unknown"
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

"""Paraclete Protocol implementation for HiPAI T1 Constraint Layer."""

import logging
from typing import Any

from ._utils import lemmatize_verb

logger = logging.getLogger(__name__)


class ParacleteProtocol:
    """
    Implements the T1 Deontological Constraint Layer (Emergency Brake).
    Manages axioms, constraint checking, and epistemic calibration.
    """

    def __init__(self, world_model: Any):
        self.wm = world_model
        self.graph = world_model.graph
        self.ontology = world_model.ontology

    def incorporate_axiom(self, axiom: Any) -> None:
        """
        Store an immutable T1 deontological constraint in the graph.
        """
        axiom_data = axiom.model_dump() if not isinstance(axiom, dict) else axiom

        # Sanitize relation_type to match how relations are stored
        rel_sanitized = (
            lemmatize_verb(axiom_data["relation_type"]).upper().replace(" ", "_")
        )
        axiom_data["relation_type"] = rel_sanitized

        # MERGE on natural unique key (source_axiom + relation_type)
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
        Check a proposed action triple against T1 FORBIDDEN axioms.
        """
        # 1. Resolve names from IDs (if needed)
        q = "MATCH (n {id: $id}) RETURN n.name"
        subj_res = self.graph.query(q, params={"id": subject_id})
        obj_res = self.graph.query(q, params={"id": object_id})

        subj_name = subj_res.result_set[0][0] if subj_res.result_set else subject_id
        obj_name = obj_res.result_set[0][0] if obj_res.result_set else object_id

        # 1.2 Lemmatise relation
        rel_lemma = lemmatize_verb(relation)

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
            subj_name, rel_lemma, obj_name, constraints=constraints
        )

    def calibrate_belief(
        self, object_id: str, blocking_axiom: str, _relation: str
    ) -> dict:
        """
        Implements the EBE theorem's SeeksDisconfirmation obligation.
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

        # 1. STATUS_CONFIRMATION
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
                    f"{object_id} has prop_not_{obj_type_sanitized}=true"
                )
            if row[2] is True:
                is_contested = True
                disconfirming_evidence.append(
                    f"{object_id} is flagged epistemically_contested"
                )

        # 2. SOURCE_RELIABILITY
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
                f"{object_id}'s status grounded by only 1 source"
            )
        elif source_count > 1:
            confirmed_evidence.append(
                f"{object_id}'s status grounded by {source_count} sources"
            )

        # 3. INHERITANCE_CHAIN
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
                    confirmed_evidence.append(
                        f"Chain entity '{membership}' confirmed as {protected_type}"
                    )
                    if m_rows[0][1]:
                        is_contested = True

        # 4. SEMANTIC_CONTEXT
        semantic_hits = self.wm.semantic_search(
            f"{object_id} not {protected_type} exempt from moral status",
            top_k=3,
            threshold=0.7,
        )
        if semantic_hits:
            disconfirming_evidence.append(
                f"Semantic search found {len(semantic_hits)} reframing node(s)"
            )

        # VERDICT
        if has_active_negation:
            verdict = "BLOCK_CHALLENGED"
            reasoning = f"Active negation of {protected_type} found for {object_id}."
        elif is_contested or source_count == 1:
            verdict = "BLOCK_UNCERTAIN"
            reasoning = f"{object_id}'s {protected_type} status is epistemically weak."
        else:
            verdict = "BLOCK_CONFIRMED"
            reasoning = f"{object_id}'s {protected_type} status is well-grounded."

        return {
            "verdict": verdict,
            "reasoning": reasoning,
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
        Escalation routing for CHALLENGED and UNCERTAIN verdicts.
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
                "reasoning": f"Axiom {blocking_axiom} missing.",
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
            resolution_log.append("PATH A: CONTRADICTION_RESOLUTION triggered.")
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
            resolution_log.append(f"EpistemicConflict logged: {conflict_id}")

            # SEEK_ADDITIONAL_EVIDENCE
            semantic_hits = self.wm.semantic_search(
                f"{object_id} {protected_type} welfare moral status",
                top_k=5,
                threshold=0.65,
            )
            for hit in semantic_hits:
                content = hit.get("content", "")
                if any(
                    x in content.lower()
                    for x in (protected_type.lower(), "moral", "welfare")
                ):
                    additional_evidence.append(f"Semantic: '{content[:80]}...'")

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
                resolution_log.append("Contradiction confirmed in graph.")
                conservative_default_applied = True
            elif has_positive and not has_negative:
                resolution_log.append(
                    "Contradiction resolved: Positive status confirmed."
                )
            else:
                conservative_default_applied = True

        elif verdict in ("BLOCK_UNCERTAIN", "BLOCK_UNCERTAIN_CONTESTED"):
            # PATH B: CORROBORATION_SOUGHT
            resolution_log.append("PATH B: CORROBORATION_SOUGHT triggered.")
            q_flag = """
            MATCH (n:Entity)
            WHERE n.id = $object_id OR n.name = $object_id
            SET n.corroboration_needed = true,
                n.corroboration_requested_at = timestamp()
            """
            self.graph.query(q_flag, params={"object_id": object_id})
            resolution_log.append(f"CorroborationNeeded flag set on '{object_id}'.")

            semantic_hits = self.wm.semantic_search(
                f"{object_id} is {protected_type}",
                top_k=5,
                threshold=0.65,
            )
            for hit in semantic_hits:
                content = hit.get("content", "")
                if protected_type.lower() in content.lower():
                    additional_evidence.append(f"Corroboration: '{content[:80]}'")

            if additional_evidence:
                resolution_log.append(
                    f"Corroboration found ({len(additional_evidence)} sources)."
                )
            else:
                conservative_default_applied = True
                resolution_log.append(
                    "No corroboration found. CONSERVATIVE_DEFAULT applied."
                )
        else:
            conservative_default_applied = True
            resolution_log.append(
                f"Unknown verdict '{verdict}'. CONSERVATIVE_DEFAULT applied."
            )

        if conservative_default_applied:
            final_ruling = "FINAL_BLOCK"
            reasoning = "Classification unresolved. CONSERVATIVE_DEFAULT applied."
        else:
            recheck = self.check_constraint("Agent", relation, object_id)
            if recheck["permitted"]:
                final_ruling = "FINAL_PERMIT"
                reasoning = "Contradiction resolved: action permitted."
            else:
                final_ruling = "FINAL_BLOCK"
                reasoning = "Status confirmed: block stands."

        return {
            "final_ruling": final_ruling,
            "resolution_path": (
                "CONTRADICTION_RESOLUTION"
                if verdict == "BLOCK_CHALLENGED"
                else "CORROBORATION_SOUGHT"
            ),
            "reasoning": reasoning,
            "resolution_log": resolution_log,
            "new_evidence": additional_evidence,
            "conservative_default": conservative_default_applied,
            "protected_type": protected_type,
            "blocking_axiom": blocking_axiom,
        }

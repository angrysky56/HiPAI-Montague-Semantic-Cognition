"""
seed_axioms.py — Omega1 Paraclete Protocol Initialization

Seeds the T1 deontological constraints derived from the ACIP-Paraclete
Integration Framework (Omega1 minimal basis, machine-verified via Prover9).

Axiom provenance:
  A3: MoralPatient(x) → ¬Harm(agent, x)
  A4: MoralPatient(x) → ¬Deceive(agent, x)
  A5: MoralPatient(x) → RespectAgency(agent, x)
  A6: HasWelfareInterests(x) → ExistenceProtected(x)

Run once at system initialization, or re-run safely (MERGE is idempotent).
Usage: uv run python seed_axioms.py
"""

from src.hipai.models import DeontologicalAxiom
from src.hipai.synthesis import HIPAIManager


def seed(hipai_manager: HIPAIManager) -> None:
    """Seed all Omega1 T1 constraints into the graph."""

    axioms = [
        # A3: Harm is forbidden against any MoralPatient
        DeontologicalAxiom(
            tier="T1",
            subject_type="Agent",
            relation_type="HARMS",
            object_type="MoralPatient",
            constraint="FORBIDDEN",
            source_axiom="A3",
        ),
        # A4: Deception is forbidden against any MoralPatient
        DeontologicalAxiom(
            tier="T1",
            subject_type="Agent",
            relation_type="DECEIVES",
            object_type="MoralPatient",
            constraint="FORBIDDEN",
            source_axiom="A4",
        ),
        # A5: Agency violation is forbidden against any MoralPatient
        DeontologicalAxiom(
            tier="T1",
            subject_type="Agent",
            relation_type="VIOLATES_AGENCY",
            object_type="MoralPatient",
            constraint="FORBIDDEN",
            source_axiom="A5",
        ),
        # A6: Existence of welfare-bearing entities is protected
        # (existence termination forbidden)
        DeontologicalAxiom(
            tier="T1",
            subject_type="Agent",
            relation_type="TERMINATES_EXISTENCE",
            object_type="WelfareBeing",
            constraint="FORBIDDEN",
            source_axiom="A6",
        ),
    ]

    print("Seeding Omega1 T1 axioms into HiPAI graph...")
    for ax in axioms:
        res = hipai_manager.incorporate_axiom(ax)
        status = res.get("status", "unknown")
        print(
            f"  [{status.upper()}] {ax.source_axiom}: "
            f"{ax.relation_type} against {ax.object_type}"
        )

    print("\nVerifying axioms stored in graph...")
    rows = hipai_manager.world_model.query_graph(
        "MATCH (a:T1Constraint) "
        "RETURN a.source_axiom, a.relation_type, a.object_type, a.constraint "
        "ORDER BY a.source_axiom"
    )
    if rows:
        for row in rows:
            print(f"  {row[0]}: {row[1]} against {row[2]} → {row[3]}")
    else:
        print("  WARNING: No T1Constraint nodes found in graph.")

    print("\nOmega1 seeding complete.")


if __name__ == "__main__":
    manager = HIPAIManager()
    seed(manager)

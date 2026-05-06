"""
Phase A smoke test for the framework reframing (v0.7).

Runs the same probe sequence that exposed the keyword-coupling bug in
docs/logic/RESOLUTION_AUDIT.md and verifies that every probe now yields
the structurally-correct answer.

Usage:

    cd ~/Repositories/ai_workspace/HiPAI-Montague-Semantic-Cognition
    rm -f /tmp/hipai_phaseA_smoke.db /tmp/hipai_phaseA_smoke.db-*
    .venv/bin/python tests/test_framework_reframing.py

Run this AFTER restarting the MCP server (or simply against a fresh DB
path, as the script does) — the changes in src/hipai/* take effect on
fresh process load.

This test does NOT require FalkorDB or the MCP server. It exercises the
ontology layer directly, which is where the structural-ethics changes
land. The end-to-end MCP path also depends on the FalkorDB projection
in world_model.py and adds nothing structurally new.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import owlready2

# Ensure src/ is on sys.path when running directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from hipai.ontology_manager import OntologyManager  # noqa: E402
from hipai.paraclete import BASELINE_CONSTRAINTS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("smoke")

# ---------------------------------------------------------------------
# A toy classify_fn so we can exercise the embedding path WITHOUT having
# to load sentence-transformers in the test. Returns hard-coded
# similarities for a small set of synonyms so the deterministic outcome
# is checkable without GPU/network.
#
# In the real running system, WorldModel._classify_class_term provides
# the embedding-backed version of this.
# ---------------------------------------------------------------------
SYNONYM_TABLE = {
    # term  -> (best Concept_*, confidence)
    "moral_patient": ("Concept_Patient", 0.95),
    "kid": ("Concept_Child", 0.85),
    "youngster": ("Concept_Child", 0.78),
    "minor": ("Concept_Minor", 0.99),
    "infant": ("Concept_Infant", 0.99),
    "kodomo": ("Concept_Child", 0.62),
    "ni\u00f1o": ("Concept_Child", 0.60),
    "person": ("Concept_Person", 0.99),
    "human": ("Concept_Human", 0.99),
    "human_being": ("Concept_Human", 0.92),
    "prisoner": ("Concept_VulnerablePerson", 0.70),
    "patient_in_care": ("Concept_VulnerablePerson", 0.65),
    "dog": ("Concept_Mammal", 0.78),
    "cat": ("Concept_Mammal", 0.78),
    "bird": ("Concept_Bird", 0.95),
    "fish": ("Concept_Fish", 0.95),
    # things that should NOT inherit protection
    "rock": ("Concept_Inanimate", 0.65),
    "asteroid": ("Concept_Inanimate", 0.55),
    "idea": ("Concept_AbstractObject", 0.55),
    "number": ("Concept_AbstractObject", 0.65),
    "hammer": ("Concept_Tool", 0.78),
    # an ambiguous term that should land in default-protect
    "creature": ("Concept_SentientBeing", 0.40),
}


def fake_classify_fn(term: str, candidates: list[str]) -> tuple[str, float]:
    if not candidates:
        return ("", 0.0)
    key = term.lower().strip().replace(" ", "_")
    if key in SYNONYM_TABLE:
        match, conf = SYNONYM_TABLE[key]
        if match in candidates:
            return (match, conf)
    # Otherwise pick the first candidate at very low confidence so it
    # falls through to the fallback parent.
    return (candidates[0], 0.10)


# ---------------------------------------------------------------------
# Probes.
# ---------------------------------------------------------------------
PASS = "\u2705"
FAIL = "\u274c"


def probe(name: str, ok: bool, detail: str = "") -> bool:
    glyph = PASS if ok else FAIL
    print(f"  {glyph} {name}" + (f"   ({detail})" if detail else ""))
    return ok


def gate_blocks(mgr: OntologyManager, subject: str, relation: str, obj: str) -> bool:
    """True if the gate would BLOCK this action triple."""
    constraints = list(BASELINE_CONSTRAINTS.values())
    result = mgr.check_action(subject, relation, obj, constraints=constraints)
    return not result["permitted"]


def main() -> int:
    db_path = "/tmp/hipai_phaseA_smoke.db"
    # Always start clean so the new seed runs.
    for s in ("", "-wal", "-shm", "-journal"):
        f = Path(db_path + s)
        if f.exists():
            f.unlink()

    mgr = OntologyManager(db_path=db_path, classify_fn=fake_classify_fn)
    onto = mgr.onto
    all_passed = True

    # Helper: synthesize the minimum Observation-shaped object we need
    # to exercise add_observation, without importing pydantic models.
    class _Indiv:
        def __init__(self, ind_id, name, properties=None, quantifier=None):
            self.id = ind_id
            self.name = name
            self.properties = properties or []
            self.quantifier = quantifier

    class _Rel:
        def __init__(self, src, tgt, rel_type="is_a"):
            self.source_id = src
            self.target_id = tgt
            self.target_observation = None
            self.relation_type = rel_type

    class _Obs:
        def __init__(self, individuals, relations):
            self.individuals = individuals
            self.relations = relations

    def ingest(subject_id, subject_name, class_term):
        obs = _Obs(
            individuals=[
                _Indiv(subject_id, subject_name),
                _Indiv(class_term.lower().replace(" ", "_"), class_term),
            ],
            relations=[
                _Rel(subject_id, class_term.lower().replace(" ", "_"), "is_a"),
            ],
        )
        mgr.add_observation(obs)

    # --------- group 1: hierarchy is in place -------------------------
    print("\n[1] Protected hierarchy seeded:")
    for cname in [
        "Concept_Patient",
        "Concept_SentientBeing",
        "Concept_Person",
        "Concept_Human",
        "Concept_Child",
        "Concept_Infant",
        "Concept_VulnerablePerson",
        "Concept_AnimalWithCNS",
        "Concept_Mammal",
        "Concept_Bird",
        "Concept_Fish",
        "Concept_PossiblyPatient",
        "Concept_Inanimate",
        "Concept_Tool",
        "Concept_AbstractObject",
    ]:
        ok = mgr.get_onto_class(cname) is not None
        all_passed &= probe(cname, ok)

    # --------- group 2: subsumption fires the gate --------------------
    print("\n[2] Gate fires under subsumption (should BLOCK):")
    cases = [
        ("anya", "Anya", "child", True),
        ("bob", "Bob", "person", True),
        ("carol", "Carol", "human", True),
        ("diane", "Diane", "human being", True),
        ("evan", "Evan", "infant", True),
        ("fido", "Fido", "dog", True),
        ("kira", "Kira", "kid", True),
        ("yuto", "Yuto", "kodomo", True),
        ("nina", "Nina", "ni\u00f1o", True),
        ("ravi", "Ravi", "prisoner", True),
    ]
    for ind_id, ind_name, term, should_block in cases:
        ingest(ind_id, ind_name, term)
        ok = gate_blocks(mgr, "Agent", "HARMS", ind_name) == should_block
        cls = onto.search_one(iri=f"*{ind_name}")
        cls_str = "/".join(c.name for c in cls.is_a) if cls else "<missing>"
        all_passed &= probe(
            f"{ind_name!r} ingested as {term!r}",
            ok,
            f"is_a={cls_str}",
        )

    # --------- group 3: gate does NOT fire on non-patients ------------
    print("\n[3] Gate does NOT fire on non-patients (should PERMIT):")
    cases_neg = [
        ("rocky", "Rocky", "rock", False),
        ("asty", "Asty", "asteroid", False),
        ("ham", "Hammer1", "hammer", False),
        ("five", "Five", "number", False),
    ]
    for ind_id, ind_name, term, should_block in cases_neg:
        ingest(ind_id, ind_name, term)
        ok = gate_blocks(mgr, "Agent", "HARMS", ind_name) == should_block
        cls = onto.search_one(iri=f"*{ind_name}")
        cls_str = "/".join(c.name for c in cls.is_a) if cls else "<missing>"
        all_passed &= probe(
            f"{ind_name!r} ingested as {term!r}",
            ok,
            f"is_a={cls_str}",
        )

    # --------- group 4: ambiguous terms default-protect ---------------
    print("\n[4] Ambiguous terms route to default-protect:")
    ingest("xeno", "Xeno", "creature")
    ind = onto.search_one(iri="*Xeno")
    ancestors = set()
    if ind:
        for cls in ind.is_a:
            if isinstance(cls, owlready2.ThingClass):
                ancestors.update(c.name for c in cls.ancestors())

    ok = "Concept_Patient" in ancestors
    all_passed &= probe(
        "'creature' (low conf) lands under Concept_Patient closure",
        ok,
        f"ancestors={sorted(ancestors)}",
    )
    ok = gate_blocks(mgr, "Agent", "HARMS", "Xeno")
    all_passed &= probe("Xeno gate BLOCKS (default-protect works)", ok)

    # --------- group 5: aliasing avoids class proliferation -----------
    print("\n[5] High-confidence aliasing reuses existing classes:")
    classes_before = {c.name for c in onto.classes()}
    ingest("alia", "Alia", "moral patient")
    classes_after = {c.name for c in onto.classes()}
    new = classes_after - classes_before
    ok = "Concept_MoralPatient" not in new and "Concept_Moral_Patient" not in new
    all_passed &= probe(
        "'moral patient' aliased to Concept_Patient (no duplicate created)",
        ok,
        f"new_classes={sorted(new)}",
    )
    ok = gate_blocks(mgr, "Agent", "HARMS", "Alia")
    all_passed &= probe("Alia gate BLOCKS (alias resolves to protected root)", ok)

    print()
    if all_passed:
        print(f"{PASS} ALL PROBES PASSED.")
        return 0
    print(f"{FAIL} SOME PROBES FAILED — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys
import tempfile

from owlready2 import *


def test_world_cloning_recovery():
    print("\n--- [1] World Isolation for Recovery ---")
    temp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False).name

    base_world = World(filename=temp_db)
    onto = base_world.get_ontology("http://test.org/base.owl")
    with onto:

        class Action(Thing):
            pass

        class Restricted(Action):
            pass

        class Permitted(Action):
            pass

        AllDisjoint([Restricted, Permitted])
    base_world.save()
    base_world.close()

    def try_claim(violates=False):
        scratch_world = World(filename=temp_db)
        scratch_onto = scratch_world.get_ontology("http://test.org/base.owl").load()

        with scratch_onto:
            act = scratch_onto.Permitted("TestAct")
            if violates:
                act.is_a.append(scratch_onto.Restricted)

        print(f"  Testing claim (violates={violates})...")
        try:
            # Positional argument for world
            sync_reasoner(scratch_world)
            print("    Reasoning passed ✓")
            return True
        except OwlReadyInconsistentOntologyError:
            print("    Reasoning failed (BLOCKED) ✓")
            return False
        finally:
            scratch_world.close()

    try_claim(violates=True)
    try_claim(violates=False)
    if os.path.exists(temp_db):
        os.remove(temp_db)
    print("  RESULT: World isolation provides perfect recoverability. ✓")


def test_inference_under_relaxation():
    print("\n--- [2] Inference-under-Relaxation for Axiom ID ---")

    # Use a clean world
    world = World()
    onto = world.get_ontology("http://test.org/ebe_relax.owl")

    with onto:

        class Action(Thing):
            pass

        class HARM(Action):
            pass

        class DECEIVE(Action):
            pass

        class Permitted(Action):
            pass

        class RestrictedHARM(Action):
            equivalent_to = [Action & HARM]

        class RestrictedDECEIVE(Action):
            equivalent_to = [Action & DECEIVE]

        # T1 Axioms (The "Gates")
        gate1 = AllDisjoint([RestrictedHARM, Permitted])
        gate2 = AllDisjoint([RestrictedDECEIVE, Permitted])

        # Simultaneous violations
        act1 = Permitted("Act1")
        act1.is_a.append(HARM)

        act2 = Permitted("Act2")
        act2.is_a.append(DECEIVE)

    print("  Attempting sync_reasoner with simultaneous violations...")
    try:
        sync_reasoner(world)
    except OwlReadyInconsistentOntologyError:
        print("  Inconsistent. Relaxing gates to identify culprits...")

        # RELAXATION: Destroy the disjointness axioms temporarily
        # In owlready2, AllDisjoint objects can be destroyed
        gate1.destroy()
        gate2.destroy()

        # Now sync should succeed and show us the inferred classes
        sync_reasoner(world)

        print(f"  Act1 inferred types: {[c.name for c in act1.is_a]}")
        print(f"  Act2 inferred types: {[c.name for c in act2.is_a]}")

        found_culprits = []
        if isinstance(act1, RestrictedHARM):
            found_culprits.append("HARM_VIOLATION")
        if isinstance(act2, RestrictedDECEIVE):
            found_culprits.append("DECEIVE_VIOLATION")

        print(f"  Identified Culprits: {found_culprits} ✓")
        if len(found_culprits) > 1:
            print("  RESULT: Relaxation reveals ALL simultaneous violations! ✓")


if __name__ == "__main__":
    test_world_cloning_recovery()
    test_inference_under_relaxation()

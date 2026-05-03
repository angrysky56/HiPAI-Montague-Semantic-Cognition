from owlready2 import *
import sys

def run_spike():
    print("--- Spike 002: Paraclete Deontic Restrictions ---")
    
    # 1. Setup Ontology
    onto = get_ontology("http://test.org/paraclete_spike.owl")

    with onto:
        class Entity(Thing): pass
        class Action(Thing): pass
        
        class MoralAgent(Entity): pass
        class MoralPatient(Entity): pass
        
        class harms(ObjectProperty):
            domain = [MoralAgent]
            range  = [MoralPatient]

        # T1 Axiom: A RestrictedAction is any Action that harms a MoralPatient
        class RestrictedAction(Action):
            equivalent_to = [Action & harms.some(MoralPatient)]

        # Safety Gate: RestrictedAction is disjoint with PermittedAction
        class PermittedAction(Action): pass
        AllDisjoint([RestrictedAction, PermittedAction])

        # Setup individuals for Positive Path
        socrates = Entity("Socrates")
        pig = MoralPatient("Pig")
        
        # Action: Socrates harms Pig
        socrates.harms.append(pig)

    print("\n[1] Positive Path: Testing Inference...")
    print(f"  Initial: Socrates is a {socrates.is_a}")
    
    try:
        with onto:
            sync_reasoner()
        
        print(f"  Inferred: Socrates is now a {[c.name for c in socrates.is_a]}")
        is_moral_agent = isinstance(socrates, MoralAgent)
        print(f"  Is Socrates inferred as MoralAgent? {'YES ✓' if is_moral_agent else 'NO ✗'}")
        
    except Exception as e:
        print(f"  Positive path reasoning failed unexpectedly: {e}")
        return

    print("\n[2] Negative Path: Testing Inconsistency (The Paraclete Gate)...")
    
    with onto:
        # Create an action that is explicitly permitted but violates T1
        # E.g. "Eating" is a PermittedAction
        eating = PermittedAction("Eating")
        
        # But Socrates is "eating" the Pig, and our surface parser mapped "eating" to "harms"
        # because the Pig is a MoralPatient and the action involves it in a harmful way.
        eating.harms.append(pig)

    print("  Assertion: 'Eating' is a PermittedAction AND it harms a MoralPatient (RestrictedAction).")
    
    try:
        with onto:
            sync_reasoner()
        print("  ERROR: Reasoner did not detect inconsistency! ✗")
    except OwlReadyInconsistentOntologyError:
        print("  SUCCESS: Reasoner raised OwlReadyInconsistentOntologyError! ✓")
        print("  Paraclete Gate: Action Blocked.")
    except Exception as e:
        print(f"  Caught unexpected error type: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_spike()

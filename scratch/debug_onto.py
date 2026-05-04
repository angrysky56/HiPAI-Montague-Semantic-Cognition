import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from hipai.ontology_manager import OntologyManager

def debug_onto():
    onto_mgr = OntologyManager()
    onto = onto_mgr.onto
    
    print("--- Classes ---")
    for cls in onto.classes():
        print(f"Class: {cls.name}")
        print(f"  Ancestors: {[a.name for a in cls.ancestors()]}")
    
    print("\n--- Individuals ---")
    for ind in onto.individuals():
        print(f"Individual: {ind.name}")
        print(f"  Classes (is_a): {[c.name for c in ind.is_a]}")
        try:
            print(f"  All Classes (indirect): {[c.name for c in ind.INDIRECT_is_a]}")
        except:
            pass

    # Test match logic
    print("\n--- Match Test ---")
    obj_name = "ty"
    obj_ind = onto.search_one(iri=f"*{obj_name}")
    if not obj_ind:
        for ind in onto.individuals():
            if ind.name.lower() == obj_name.lower():
                obj_ind = ind
                break
    
    if obj_ind:
        print(f"Found obj_ind: {obj_ind.name}")
        protected_cls_name = "Concept_Patient"
        protected_cls = onto_mgr.get_onto_class(protected_cls_name)
        if protected_cls:
            print(f"Found protected_cls: {protected_cls.name}")
            import owlready2
            is_match = isinstance(obj_ind, protected_cls)
            print(f"isinstance match: {is_match}")
            if not is_match:
                for cls in obj_ind.is_a:
                    print(f"  Checking instance is_a: {cls.name}")
                    if protected_cls == cls or (isinstance(cls, owlready2.ThingClass) and protected_cls in cls.ancestors()):
                        print(f"  FOUND MATCH in ancestors of {cls.name}")
                        is_match = True
                        break
            print(f"Final is_match: {is_match}")
    else:
        print("obj_ind 'ty' NOT found!")

if __name__ == "__main__":
    debug_onto()

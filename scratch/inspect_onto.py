import sys
from pathlib import Path

import owlready2
from owlready2 import World

db_path = str(Path("world.db").resolve())
world = World(filename=db_path)
onto = world.get_ontology("http://hipai.org/onto.owl").load()

print(f"Classes: {[c.name for c in onto.classes()]}")
print(f"Individuals: {[i.name for i in onto.individuals()]}")

dave = onto.search_one(iri="*dave")
if dave:
    print(f"Dave is_a: {[c.name for c in dave.is_a]}")
else:
    print("Dave not found")

cmp = getattr(onto, "Concept_Moral_Patient", None)
if cmp:
    print(f"Concept_Moral_Patient ancestors: {[a.name for a in cmp.ancestors()]}")
    if dave:
        print(f"isinstance(dave, Concept_Moral_Patient): {isinstance(dave, cmp)}")
else:
    print("Concept_Moral_Patient not found")

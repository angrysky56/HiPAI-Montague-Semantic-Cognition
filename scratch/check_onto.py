
import owlready2
from pathlib import Path

db_path = "world.db"
if Path(db_path).exists():
    world = owlready2.World(filename=str(Path(db_path).resolve()))
    onto = world.get_ontology("http://hipai.org/ontology").load()
    print(f"Classes: {[c.name for c in onto.classes()]}")
    print(f"Individuals: {[i.name for i in onto.individuals()]}")
    world.close()
else:
    print("world.db not found")

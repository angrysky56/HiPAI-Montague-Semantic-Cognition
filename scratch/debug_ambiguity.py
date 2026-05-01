from hipai.synthesis import HIPAIManager
from hipai.exceptions import AmbiguityDetectedError

manager = HIPAIManager(graph_name="debug_ambiguity")
manager.world_model.clear_graph()

try:
    manager.add_belief("Socrates is a man")
except AmbiguityDetectedError as e:
    print(f"AMBIGUITY: {[p['type'] for p in e.possible_parses]}")
except Exception as e:
    print(f"ERROR: {type(e)} {e}")
else:
    print("SUCCESS")

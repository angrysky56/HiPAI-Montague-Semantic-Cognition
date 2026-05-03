
from hipai.synthesis import HIPAIManager


def test_socrates_debug():
    manager = HIPAIManager()
    manager.world_model.clear_graph()
    
    print("Adding Socrates is a man...")
    res = manager.add_belief("Socrates is a man")
    print(f"Result: {res['status']}")
    
    state = manager.get_current_state()
    print(f"Nodes: {state['nodes']}")

if __name__ == "__main__":
    test_socrates_debug()

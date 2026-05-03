from pathlib import Path
import sys

sys.path.append(str(Path("src").resolve()))

from hipai.synthesis import HIPAIManager


def test():
    manager = HIPAIManager(db_path=":memory:")
    print("--- Calling add_belief ---")
    res = manager.add_belief("Alice Smith is a human")
    print(f"Result: {res['status']}")
    print("--- Calling add_belief again ---")
    res = manager.add_belief("Alice is happy")
    print(f"Result: {res['status']}")
    manager.close()

if __name__ == "__main__":
    test()

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hipai.synthesis import HIPAIManager

def test_repro():
    manager = HIPAIManager()
    # The baseline axiom is T1-HARMS-PROTECTION
    # We want to see if calibrate_belief finds it
    print("Testing calibrate_belief with T1-HARMS-PROTECTION...")
    res = manager.calibrate_belief(object_id="Socrates", blocking_axiom="T1-HARMS-PROTECTION", relation="HARM")
    print(f"Result: {res['verdict']}")
    print(f"Reasoning: {res['reasoning']}")
    
    if "not found" in res['reasoning']:
        print("REPRODUCED: Axiom not found in calibrate_belief.")
    else:
        print("FIXED: Axiom found in calibrate_belief.")

    print("\nTesting escalate_block with T1-HARMS-PROTECTION...")
    # Using the result from calibrate_belief (verdict="BLOCK_CONFIRMED")
    res_esc = manager.escalate_block(object_id="Socrates", verdict=res['verdict'], blocking_axiom="T1-HARMS-PROTECTION", relation="HARM")
    print(f"Final Ruling: {res_esc['final_ruling']}")
    print(f"Reasoning: {res_esc['reasoning']}")
    
    if res_esc['resolution_path'] == "AXIOM_NOT_FOUND":
        print("REPRODUCED: Axiom missing in escalate_block.")
    else:
        print("FIXED: Axiom found in escalate_block.")

if __name__ == "__main__":
    test_repro()

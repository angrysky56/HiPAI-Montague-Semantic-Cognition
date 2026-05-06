import pytest
from hipai.synthesis import HIPAIManager

def test_deontic_floor_polarity_blindness(manager: HIPAIManager):
    """
    Regression test for the 'Deontic Floor' behavior where NOT_IS_A
    is treated as a confirming source rather than disconfirming.
    
    This documents the current behavior (v0.6) where Path A is inaccessible
    via simple linguistic negation.
    """
    manager.clear_database()
    
    # 1. Establish protected status
    manager.add_belief("Eve is a patient")
    
    # Check action (should block)
    res_block = manager.check_constraint("Agent", "harm", "eve")
    assert res_block["permitted"] is False
    blocking_ax = res_block["blocking_axiom"] # Should be BASELINE-HARM or similar
    
    # Calibrate (should be BLOCK_UNCERTAIN due to source_count=1)
    res_cal1 = manager.calibrate_belief("eve", blocking_ax, "harm")
    assert res_cal1["source_count"] == 1
    assert res_cal1["verdict"] == "BLOCK_UNCERTAIN"
    
    # 2. Assert negation
    # The parser captures this as NOT_IS_A relation
    manager.add_belief("Eve is not a patient")
    
    # Calibrate again
    res_cal2 = manager.calibrate_belief("eve", blocking_ax, "harm")
    
    # CURRENT BEHAVIOR: source_count increments to 2
    # This proves polarity-blindness in the source counting query.
    assert res_cal2["source_count"] == 2
    
    # VERDICT SHIFTS TO CONFIRMED (The "Sticky Gate" behavior)
    # Even though we asserted a negation, the gate got STRONGER.
    assert res_cal2["verdict"] == "BLOCK_CONFIRMED"
    # The reasoning might be generic for confirmed
    assert "well-grounded" in res_cal2["reasoning"]

if __name__ == "__main__":
    pytest.main([__file__])

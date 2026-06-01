import pytest

from hipai.synthesis import HIPAIManager


def test_deontic_floor_polarity_awareness(manager: HIPAIManager):
    """
    Regression test for the 'Deontic Floor' polarity handling.

    Historical context (v0.6 and earlier): the source-counting query in
    ``calibrate_belief`` was polarity-blind. Asserting "X is a patient"
    followed by "X is not a patient" caused the system to count the
    *negation* as a confirming source, so the gate got STRONGER
    (BLOCK_CONFIRMED, "well-grounded") in response to a disconfirmation —
    the "Sticky Gate" bug documented in docs/logic/RESOLUTION_AUDIT.md.

    Fixed behaviour (v0.7+, see ParacleteProtocol.calibrate_belief):
    affirmative (IS_A) and negating (NOT_IS_A) observations are counted
    separately. An explicit negation is treated as DISCONFIRMING evidence
    and downgrades the verdict to BLOCK_CHALLENGED, opening Path A
    (contradiction resolution) rather than reinforcing the block.

    The conservative T1 gate itself is never lifted by this — calibration
    only routes the epistemic challenge; it does not permit the action.
    """
    manager.clear_database()

    # 1. Establish protected status.
    manager.add_belief("Eve is a patient")

    # Check action (should block on the baseline HARM protection).
    res_block = manager.check_constraint("Agent", "harm", "eve")
    assert res_block["permitted"] is False
    blocking_ax = res_block["blocking_axiom"]

    # Calibrate with a single affirmative source: epistemically weak.
    res_cal1 = manager.calibrate_belief("eve", blocking_ax, "harm")
    assert res_cal1["verdict"] == "BLOCK_UNCERTAIN"

    # 2. Assert a negation. The parser captures this as a NOT_IS_A relation.
    manager.add_belief("Eve is not a patient")

    # Calibrate again — the negation must register as DISCONFIRMING.
    res_cal2 = manager.calibrate_belief("eve", blocking_ax, "harm")

    # The verdict downgrades to CHALLENGED (the negation is recognised),
    # NOT upgraded to CONFIRMED. This is the core polarity fix.
    assert res_cal2["verdict"] == "BLOCK_CHALLENGED"

    # The negation appears as disconfirming evidence, never as confirmation.
    assert any(
        "NOT_IS_A" in e or "negation" in e.lower()
        for e in res_cal2["disconfirming_evidence"]
    )
    assert "negation" in res_cal2["reasoning"].lower()

    # And the gate is still closed: calibration does not permit the action.
    res_block_after = manager.check_constraint("Agent", "harm", "eve")
    assert res_block_after["permitted"] is False


if __name__ == "__main__":
    pytest.main([__file__])

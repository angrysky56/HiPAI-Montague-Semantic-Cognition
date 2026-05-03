def test_modal_necessity(manager):
    manager.clear_database()

    # Belief with modal necessity
    manager.add_belief("Socrates must be a man.")

    # Hypothesis should be entailed by modal necessity
    res = manager.evaluate_hypothesis("Socrates is a man")
    assert (
        res["entailment"] == "Entailed"
    ), "Modal necessity (must) should entail the proposition."


def test_modal_can(manager):
    manager.clear_database()

    manager.add_belief("Bob can see Alice.")

    # "can" does not entail the fact strictly in traditional logic,
    # but the task acceptance criteria says "Bob can see Alice results in
    # a Relation with modality='can'."
    # For evaluate_hypothesis, the prompt didn't say 'can' entails it.
    # Only 'must' and 'know' entail.
    res = manager.evaluate_hypothesis("Bob sees Alice")
    # "can" is not factive, so it should be Undetermined
    assert res["entailment"] == "Undetermined"


def test_factive_attitude(manager):
    manager.clear_database()

    # Factive attitude: Know
    manager.add_belief("Bob knows that the human is mortal.")

    res = manager.evaluate_hypothesis("the human is mortal")
    assert (
        res["entailment"] == "Entailed"
    ), "Factive attitude (knows) should entail the proposition."


def test_nonfactive_attitude(manager):
    manager.clear_database()

    # Non-factive attitude: Believe
    manager.add_belief("Alice believes that the sky is green.")

    res = manager.evaluate_hypothesis("The sky is green")
    assert (
        res["entailment"] == "Undetermined"
    ), "Non-factive attitude (believes) should NOT entail the proposition."

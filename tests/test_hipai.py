import json

from hipai.synthesis import HIPAIManager


def main():
    manager = HIPAIManager()

    print("--- Adding Beliefs ---")
    print(manager.add_belief("Socrates is a man."))
    print(manager.add_belief("All men are mortal."))

    print("\n--- Current State ---")
    state = manager.get_current_state()
    print(json.dumps(state, indent=2))

    print("\n--- Evaluating Hypothesis ---")
    hypothesis = "Socrates is mortal."
    print(f"Hypothesis: {hypothesis}")
    result = manager.evaluate_hypothesis(hypothesis)
    print("Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

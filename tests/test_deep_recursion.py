

def test_deep_recursion(manager):
    # Alice believes Bob thinks Charlie is happy
    text = "Alice believes Bob thinks Charlie is happy"
    print(f"\nProcessing: {text}")

    result = manager.add_belief(text)
    print(f"Result: {result['status']}")

    # Verify in OWL
    with manager.world_model.ontology.onto:
        print(
            f"Classes in onto: "
            f"{[c.name for c in manager.world_model.ontology.onto.classes()]}"
        )
        print(
            f"Individuals in onto: "
            f"{[i.name for i in manager.world_model.ontology.onto.individuals()]}"
        )

        # Use more flexible search
        alice = manager.world_model.ontology.onto.search_one(iri="*alice*")
        bob = manager.world_model.ontology.onto.search_one(iri="*bob*")
        charlie = manager.world_model.ontology.onto.search_one(iri="*charlie*")

        print(
            f"\nFound individuals: "
            f"Alice={bool(alice)} ({alice.name if alice else 'N/A'}), "
            f"Bob={bool(bob)}, Charlie={bool(charlie)}"
        )

        # Check Alice's belief
        obs_class = manager.world_model.ontology.onto.Observation
        print(f"Observation class found: {bool(obs_class)}")

        if obs_class and alice:
            # Look for observations where source is Alice
            beliefs = [r for r in obs_class.instances() if r.source == alice]
        else:
            beliefs = []
        print(f"Alice's nested observations count: {len(beliefs)}")

        for b in beliefs:
            print(f"Alice believes: {b.nested_observation}")
            if b.nested_observation:
                inner = b.nested_observation
                print(f"  Inner source: {inner.source}")
                print(f"  Inner relation: {inner.relation_type}")
                print(f"  Inner nested: {inner.nested_observation}")

                if inner.nested_observation:
                    deepest = inner.nested_observation
                    print(f"    Deepest source: {deepest.source}")
                    print(f"    Deepest target: {deepest.target}")

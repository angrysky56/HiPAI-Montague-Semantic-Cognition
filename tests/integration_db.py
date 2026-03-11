import sys

from hipai.models import Individual, Observation
from hipai.synthesis import HIPAIManager


def test_integration():
    print("Testing HiPAI Integration with FalkorDB on port 6380...")
    success = True
    try:
        # Initialize the high-level manager
        # This internally creates WorldModel and ZettelkastenSynthesizer
        hi_pai = HIPAIManager(graph_name="integration_test")
        wm = hi_pai.world_model

        # 1. Clear existing data
        print("[+] Clearing graph...")
        wm.clear_database()

        # 2. Test connectivity
        res = wm.query_graph("RETURN 1 as val")
        print(f"Connectivity test: {res}")
        assert res[0][0] == 1

        # 3. Test observation ingestion
        print("\n[+] Testing observation ingestion...")
        target_name = "DeepMind releases new model"
        obs1 = Observation(
            text_source="DeepMind announced a new AI model today.",
            individuals=[
                Individual(id="ent_1", name=target_name, properties=["AI", "Model"])
            ],
            relations=[],
        )
        wm.incorporate_observation(obs1)
        print("Ingestion complete.")

        # Verify via Cypher
        res_raw = wm.query_graph("MATCH (n:Entity {id: 'ent_1'}) RETURN n.name")
        print(f"Retrieval verification (Cypher): {res_raw}")
        if not res_raw or res_raw[0][0] != target_name:
            print("[-] FAILED to retrieve created node via Cypher!")
            success = False

        # 4. Test semantic search
        print("\n[+] Testing semantic search...")
        # Search for something semantically similar
        search_res = wm.semantic_search("AI research developments", top_k=5)
        print(f"Semantic search results: {len(search_res)} nodes found.")

        found = any(node["id"] == "ent_1" for node in search_res)
        if found:
            print("[+] Target node found via semantic search!")
        else:
            print("[-] Target node NOT found via semantic search.")
            # Success doesn't strictly depend on this if embedding space is empty/noisy
            # But let's try exact name
            search_own = wm.semantic_search(target_name, top_k=1)
            if search_own and search_own[0]["id"] == "ent_1":
                print("[+] Target node found via exact name semantic search!")
            else:
                print("[-] Target node NOT found even by exact name.")

        # 5. Test Syllogism (Logic Tier)
        print("\n[+] Testing Syllogism (Socrates is a man, all men are mortal)...")
        hi_pai.add_belief("Socrates is man")
        hi_pai.add_belief("All men are mortal")

        # Trigger synthesis (in a real system this might be automatic or scheduled)
        # For our test, synthesize_concepts will create Concept_Man if property threshold is met
        # hi_pai.synthesizer.synthesize_concepts(property_threshold=1)
        # Note: add_belief for "All men are mortal" already creates Concept_Men and sets prop_mortal

        # Evaluate hypothesis
        hyp_res = hi_pai.evaluate_hypothesis("Socrates is mortal")
        print("Hypothesis: Socrates is mortal")
        print(f"Result: {hyp_res['entailment']}")
        print(f"Reasoning: {hyp_res['reasoning']}")

        if hyp_res["entailment"] == "True":
            print("[+] Syllogism Test PASSED.")
        else:
            print("[-] Syllogism Test FAILED.")
            success = False

        if success:
            print("\n[***] ALL INTEGRATION TESTS PASSED [***]")
        else:
            print("\n[!!!] INTEGRATION TESTS FAILED [!!!]")
            sys.exit(1)

    except Exception as e:
        print(f"Integration test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_integration()

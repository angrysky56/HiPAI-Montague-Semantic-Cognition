"""Module for MCP server integration with HiPAI."""

import json

from mcp.server.fastmcp import FastMCP

from hipai.models import Observation
from hipai.synthesis import HIPAIManager

# Initialize FastMCP Server
mcp = FastMCP("HiPAI Server")

# Initialize HIPAIManager
# This instance manages both WorldModel and Synthesizer
hi_pai = HIPAIManager(graph_name="hipai_world")


@mcp.tool()
async def add_belief(text: str) -> str:
    """Add a belief or fact to the system in natural language
    (e.g., 'Socrates is a man')."""
    try:
        res = hi_pai.add_belief(text)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"Error adding belief: {e!s}"


@mcp.tool()
async def evaluate_hypothesis(hypothesis: str) -> str:
    """Evaluate a hypothesis against the current knowledge in the graph."""
    try:
        res = hi_pai.evaluate_hypothesis(hypothesis)
        return (
            f"Entailment: {res['entailment']}\n"
            f"Reasoning: {res['reasoning']}\n"
            f"Confidence: {res['confidence']}"
        )
    except Exception as e:
        return f"Error evaluating hypothesis: {e!s}"


@mcp.tool()
async def query_graph(cypher: str) -> str:
    """Executes a Cypher query against the HiPAI Graph Database (World Model)."""
    try:
        results = hi_pai.world_model.query_graph(cypher)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error executing query: {e!s}"


@mcp.tool()
async def synthesize_concepts(property_threshold: int = 1) -> str:
    """
    Runs the Zettelkasten Synthesis Engine to generate Structure Notes (Concepts)
    based on common properties among Content Nodes (Entities).
    """
    try:
        created = hi_pai.synthesizer.synthesize_concepts(
            property_threshold=property_threshold
        )
        return f"Synthesized Concepts: {', '.join(created) if created else 'None'}"
    except Exception as e:
        return f"Error synthesizing concepts: {e!s}"


@mcp.tool()
async def vector_synthesize_concepts(n_clusters: int = 2) -> str:
    """
    Run vector-based KMeans clustering to discover latent Concepts in the latent space.
    """
    try:
        created = hi_pai.synthesizer.vector_synthesize_concepts(n_clusters=n_clusters)
        return (
            f"Synthesized Latent Concepts: {', '.join(created) if created else 'None'}"
        )
    except Exception as e:
        return f"Error synthesizing vector concepts: {e!s}"


@mcp.tool()
async def synthesize_domains(concept_threshold: int = 1) -> str:
    """
    Runs the Zettelkasten Synthesis Engine to generate Main Structure Notes (Domains)
    by clustering related Concepts.
    """
    try:
        created = hi_pai.synthesizer.synthesize_domains(
            concept_threshold=concept_threshold
        )
        return f"Synthesized Domains: {', '.join(created) if created else 'None'}"
    except Exception as e:
        return f"Error synthesizing domains: {e!s}"


@mcp.tool()
async def ingest_observation(
    text_source: str, individuals: list[dict], relations: list[dict] | None = None
) -> str:
    """
    Ingest a cognitive observation into the World Model.

    Args:
        text_source: The original natural language sentence.
        individuals: List of individuals. Each dict needs 'name', optionally
            'id', 'properties'.
        relations: List of relations. Each dict needs 'source_id',
            'target_id', 'relation_type'.
    """
    try:
        if relations is None:
            relations = []
        obs_dict = {
            "text_source": text_source,
            "individuals": individuals,
            "relations": relations,
        }
        obs = Observation(**obs_dict)
        hi_pai.world_model.incorporate_observation(obs)
        return "Observation successfully ingested."
    except Exception as e:
        return f"Error ingesting observation: {e!s}"


@mcp.tool()
async def semantic_search(
    query_text: str, top_k: int = 5, label: str = "Entity"
) -> str:
    """Search for nodes semantically related using vector embeddings."""
    try:
        results = hi_pai.world_model.semantic_search(
            query_text, top_k=top_k, label=label
        )
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error executing semantic search: {e!s}"


@mcp.tool()
async def clear_graph() -> str:
    """Clears the HiPAI Graph Database."""
    try:
        hi_pai.clear_database()
        return "Graph database cleared."
    except Exception as e:
        return f"Error clearing graph: {e!s}"


@mcp.tool()
async def get_current_state() -> str:
    """Returns a snapshot of the current state of the World Model (nodes and edges)."""
    try:
        state = hi_pai.get_current_state()
        return json.dumps(state, indent=2)
    except Exception as e:
        return f"Error getting state: {e!s}"


if __name__ == "__main__":
    mcp.run()

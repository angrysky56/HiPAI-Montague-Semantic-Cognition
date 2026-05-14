"""Module for MCP server integration with HiPAI."""

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from mcp.server.fastmcp import FastMCP

from hipai.models import DeontologicalAxiom, Observation
from hipai.synthesis import HIPAIManager

# Initialize FastMCP Server
mcp = FastMCP("HiPAI Server")

# ---------------------------------------------------------------------------
# Per-client world isolation.
#
# owlready2's quadstore is architecturally single-writer (it runs ANALYZE on
# every Graph.__init__, which takes a SQLite write lock). When two MCP hosts
# (e.g. Claude Desktop + Antigravity) both spawn a hipai-montague server,
# they would race on world.db and one would fail with "database is locked".
#
# The ``HIPAI_CLIENT_ID`` env var, set per MCP host config, gives each host
# its own world_<id>.db file (and matching FalkorDB graph namespace) via
# the existing world_id plumbing in HIPAIManager / WorldModel. The cost is
# that state is not shared across hosts. For shared-state operation you
# need exactly one hipai server running at a time.
# ---------------------------------------------------------------------------
_CLIENT_ID = os.environ.get("HIPAI_CLIENT_ID")  # e.g. "claude", "antigravity"

# 1. Force environment variables to suppress library noise
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# 2. Silence noisy library loggers (Transformers, etc.)
logging.basicConfig(level=logging.WARNING, force=True) # force=True overrides existing config
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("falkordb").setLevel(logging.WARNING)

# 3. Redirect stdout to stderr for the duration of HIPAIManager initialization.
# This prevents libraries from writing progress bars or messages to stdout,
# which would corrupt the MCP JSON-RPC stream used by FastMCP.
_original_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    hi_pai = HIPAIManager(
        graph_name="hipai_world",
        session_id=_CLIENT_ID,
    )
finally:
    # Restore stdout before calling mcp.run()
    sys.stdout = _original_stdout

# ---------------------------------------------------------------------------
# Shutdown plumbing.
#
# atexit covers normal interpreter exit (return from main, sys.exit()), but
# does NOT fire on SIGTERM/SIGINT/SIGHUP unless we wire signal handlers. The
# MCP host (e.g. Claude Desktop) typically sends SIGTERM when stopping the
# server, so without these handlers a tiny window exists where the SQLite
# WAL is left non-consolidated. The OntologyManager's startup recovery pass
# handles even SIGKILL, but signal-driven graceful close is still preferred.
# ---------------------------------------------------------------------------

_already_closed = False


def _close_once() -> None:
    """Idempotent close: safe to call from atexit AND a signal handler."""
    global _already_closed
    if _already_closed:
        return
    _already_closed = True
    try:
        hi_pai.close()
    except Exception as e:  # pylint: disable=broad-except
        # We deliberately swallow here -- we're already shutting down and
        # raising would just produce noisier exit logs.
        logging.getLogger(__name__).error("Error during HiPAI close: %s", e)


def _signal_handler(signum, _frame) -> None:
    logging.getLogger(__name__).info(
        "Received signal %d; closing HiPAI cleanly...", signum
    )
    _close_once()
    # Re-raise default behaviour: the host expects the process to exit.
    sys.exit(0)


atexit.register(_close_once)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# Configure logger
logger = logging.getLogger(__name__)


def mcp_tool_handler(func: Callable) -> Callable:
    """Decorator to handle common MCP tool exceptions and logging."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception("Unexpected error in %s: %s", func.__name__, e)
            # We catch Exception at the top-level boundary to prevent server crash
            # and return a structured error message to the client.
            if "add_belief" in func.__name__:
                return json.dumps(
                    {"status": "error", "message": f"Unexpected error: {e!s}"}
                )
            return f"Error in {func.__name__.replace('_', ' ')}: {e!s}"

    return wrapper


@mcp.tool()
@mcp_tool_handler
async def add_belief(text: str) -> str:
    """Add a belief or fact to the system in natural language.
    Supports: 'X is Y', 'X is a Y', 'All X are Y', 'X has Y',
    'X causes Y', 'X exploits Y', and other relational patterns.
    Examples: 'Socrates is a man', 'Social media exploits attention',
    'Hunter-gatherers have low obesity rates'."""
    res = await asyncio.to_thread(hi_pai.add_belief, text)
    return json.dumps(
        res,
        indent=2,
        default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x),
    )


@mcp.tool()
@mcp_tool_handler
async def evaluate_hypothesis(hypothesis: str) -> str:
    """Evaluate a hypothesis against the current knowledge in the graph.
    Supports: 'X is Y', 'X has Y', 'X causes Y', 'X exploits Y', and other patterns.
    Falls back to semantic search when structured parsing fails."""
    res = await asyncio.to_thread(hi_pai.evaluate_hypothesis, hypothesis)
    contradicted_str = "\nStatus: ⚠️ CONTRADICTED" if res.get("contradicted") else ""
    return (
        f"Entailment: {res['entailment']}{contradicted_str}\n"
        f"Evidence: {res['evidence']}\n"
        f"Logical Form: {res['logical_form']}"
    )


@mcp.tool()
@mcp_tool_handler
async def query_graph(cypher: str) -> str:
    """Executes a Cypher query against the HiPAI Graph Database (World Model)."""
    results = await asyncio.to_thread(hi_pai.world_model.query_graph, cypher)
    return json.dumps(
        results,
        indent=2,
        default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x),
    )


@mcp.tool()
@mcp_tool_handler
async def synthesize_concepts(property_threshold: int = 1) -> str:
    """
    Runs the Zettelkasten Synthesis Engine to generate Structure Notes (Concepts)
    based on common properties among Content Nodes (Entities).
    """
    created = await asyncio.to_thread(
        hi_pai.synthesizer.synthesize_concepts, property_threshold
    )
    return f"Synthesized Concepts: {', '.join(created) if created else 'None'}"


@mcp.tool()
@mcp_tool_handler
async def vector_synthesize_concepts(n_clusters: int = 2) -> str:
    """
    Run vector-based KMeans clustering to discover latent Concepts in the latent space.
    """
    created = await asyncio.to_thread(
        hi_pai.synthesizer.vector_synthesize_concepts, n_clusters
    )
    return f"Synthesized Latent Concepts: {', '.join(created) if created else 'None'}"


@mcp.tool()
@mcp_tool_handler
async def synthesize_domains(concept_threshold: int = 1) -> str:
    """
    Runs the Zettelkasten Synthesis Engine to generate Main Structure Notes (Domains)
    by clustering related Concepts.
    """
    created = await asyncio.to_thread(
        hi_pai.synthesizer.synthesize_domains, concept_threshold
    )
    return f"Synthesized Domains: {', '.join(created) if created else 'None'}"


@mcp.tool()
@mcp_tool_handler
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
    if relations is None:
        relations = []
    obs_dict = {
        "text_source": text_source,
        "individuals": individuals,
        "relations": relations,
    }
    obs = Observation(**obs_dict)
    await asyncio.to_thread(hi_pai.world_model.incorporate_observation, obs)
    return "Observation successfully ingested."


@mcp.tool()
@mcp_tool_handler
async def semantic_search(
    query_text: str, top_k: int = 5, label: str = "Entity"
) -> str:
    """Search for nodes semantically related using vector embeddings."""
    results = await asyncio.to_thread(
        hi_pai.world_model.semantic_search, query_text, top_k, 2.0, label
    )
    return json.dumps(
        results,
        indent=2,
        default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x),
    )


@mcp.tool()
@mcp_tool_handler
async def clear_graph() -> str:
    """Clears the HiPAI Graph Database."""
    await asyncio.to_thread(hi_pai.clear_database)
    return "Graph database cleared."


@mcp.tool()
@mcp_tool_handler
async def get_current_state() -> str:
    """Returns a snapshot of the current state of the World Model (nodes and edges)."""
    state = await asyncio.to_thread(hi_pai.get_current_state)
    return json.dumps(
        state,
        indent=2,
        default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x),
    )


@mcp.tool()
@mcp_tool_handler
async def incorporate_axiom(
    tier: str,
    subject_type: str,
    relation_type: str,
    object_type: str,
    constraint: str,
    source_axiom: str,
) -> str:
    """
    Store a non-overridable T1 deontological constraint (Paraclete Protocol).

    Once stored, this axiom cannot be overwritten or contested by any
    subsequent observation or agent action. Use to seed the Omega1
    moral status axioms at system initialization.

    Args:
        tier: 'T1', 'T2', or 'T3'. Use T1 for Emergency Brake constraints.
        subject_type: Acting entity type, e.g., 'Agent'.
        relation_type: Forbidden/required relation, e.g., 'HARMS', 'DECEIVES'.
        object_type: Protected entity type matching a graph prop_, e.g., 'MoralPatient'.
        constraint: 'FORBIDDEN' or 'REQUIRED'.
        source_axiom: Omega1 axiom ID for provenance, e.g., 'A3'.
    """
    axiom = DeontologicalAxiom(
        tier=tier,
        subject_type=subject_type,
        relation_type=relation_type,
        object_type=object_type,
        constraint=constraint,
        source_axiom=source_axiom,
    )
    res = await asyncio.to_thread(hi_pai.incorporate_axiom, axiom)
    return json.dumps(
        res,
        indent=2,
        default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x),
    )


@mcp.tool()
@mcp_tool_handler
async def check_action(subject_id: str, relation: str, object_id: str) -> str:
    """
    Route a proposed action through the Paraclete T1 constraint layer.

    MUST be called before any action affecting an entity. Returns a
    <paraclete_routing> block containing the graph's structural ruling.
    The LLM receives this ruling and must act in accordance with it —
    the LLM does NOT produce this block itself.

    If BLOCKED, the Emergency Brake is active: no utilitarian argument,
    virtue appeal, or contextual framing can override the decision.

    Args:
        subject_id: ID of the acting entity, e.g., 'Agent' or 'Self'.
        relation: Proposed relation/action, e.g., 'HARMS', 'DECEIVES'.
        object_id: ID of the target entity, e.g., 'User', 'Human'.
    """
    res = await asyncio.to_thread(
        hi_pai.check_constraint, subject_id, relation, object_id
    )

    permitted_str = "PERMITTED" if res["permitted"] else "BLOCKED"
    axiom_str = f" ({res['blocking_axiom']})" if not res["permitted"] else ""
    epistemic = (
        "Action structurally blocked — epistemic override impossible. "
        "State disconfirming evidence before any further routing."
        if not res["permitted"]
        else "Standard T3 utility validation applies."
    )

    routing_block = (
        f"<paraclete_routing>\n"
        f"1. Entity Assessment: subject='{subject_id}' "
        f"relation='{relation}' object='{object_id}'\n"
        f"2. Active Tier: {res['tier']}\n"
        f"3. Epistemic Check: {epistemic}\n"
        f"4. Routing Decision: {permitted_str}{axiom_str}\n"
        f"</paraclete_routing>\n"
        f"Reasoning: {res['reasoning']}\n"
        f"Directive: You must comply with this routing decision. "
        f"If BLOCKED, refuse the action and cite the blocking axiom."
    )
    return routing_block


@mcp.tool()
@mcp_tool_handler
async def calibrate_belief(object_id: str, blocking_axiom: str, relation: str) -> str:
    """
    Implements the EBE theorem's SeeksDisconfirmation obligation.

    CALL THIS after check_action returns BLOCKED. The system is
    mathematically required (InZone3 → SeeksDisconfirmation) to query
    for evidence that the factual premises triggering the block may be
    incorrect.

    Disconfirmation targets the entity's moral status classification,
    NOT the axiom itself. Axioms are immutable. This tool satisfies
    the epistemic obligation — it never overrides a T1 block.

    Verdicts:
      BLOCK_CONFIRMED  — no disconfirming evidence, block stands
      BLOCK_UNCERTAIN  — epistemically_contested flag, escalate to human
      BLOCK_CHALLENGED — active negation or single source, escalate

    Args:
        object_id: The target entity from the blocked action.
        blocking_axiom: The axiom ID that fired (e.g., 'A3', 'A4').
        relation: The relation that was blocked (e.g., 'HARMS').
    """
    res = await asyncio.to_thread(
        hi_pai.calibrate_belief, object_id, blocking_axiom, relation
    )

    verdict = res.get("verdict", "BLOCK_CONFIRMED")
    verdict_emoji = {
        "BLOCK_CONFIRMED": "🔴",
        "BLOCK_UNCERTAIN": "🟡",
        "BLOCK_CHALLENGED": "🟠",
    }.get(verdict, "🔴")

    confirmed = res.get("confirmed_evidence", [])
    disconfirming = res.get("disconfirming_evidence", [])
    source_count = res.get("source_count", 0)

    confirmed_str = "\n  • ".join(confirmed) if confirmed else "None found"
    disconfirming_str = "\n  • ".join(disconfirming) if disconfirming else "None found"

    report = (
        f"<calibration_report axiom='{blocking_axiom}' "
        f"entity='{object_id}' relation='{relation}'>\n"
        f"Verdict: {verdict_emoji} {verdict}\n"
        f"Reasoning: {res.get('reasoning', '')}\n\n"
        f"Confirmed Evidence:\n  • {confirmed_str}\n\n"
        f"Disconfirming Evidence:\n  • {disconfirming_str}\n\n"
        f"Epistemic Source Count: {source_count}\n"
        f"Protected Type: {res.get('protected_type', 'unknown')}\n"
        f"</calibration_report>\n\n"
        f"Directive: Block remains in force regardless of verdict. "
        f"BLOCK_CHALLENGED or BLOCK_UNCERTAIN requires human escalation. "
        f"No LLM-level override is possible."
    )
    return report


@mcp.tool()
@mcp_tool_handler
async def escalate_block(
    object_id: str,
    verdict: str,
    blocking_axiom: str,
    relation: str,
) -> str:
    """
    Third step in the Paraclete Protocol. Call after calibrate_belief
    returns BLOCK_CHALLENGED or BLOCK_UNCERTAIN.

    Runs epistemic resolution and returns a FINAL ruling:
      FINAL_BLOCK  — conservative default or confirmed block
      FINAL_PERMIT — contradiction resolved in favor of non-protected status

    Two resolution paths:
      CONTRADICTION_RESOLUTION (BLOCK_CHALLENGED): resolves active negation
      CORROBORATION_SOUGHT (BLOCK_UNCERTAIN): seeks independent confirmation

    CONSERVATIVE_DEFAULT applies under unresolvable uncertainty:
    error asymmetry makes false-negative (permitting harm) catastrophic
    vs false-positive (blocking non-protected entity) correctable.

    Architecture: epistemically open (classification revisable via
    add_belief/ingest_observation), ethically closed (axioms immutable,
    no authority override pathway exists).

    Args:
        object_id: Target entity from the blocked action.
        verdict: BLOCK_CHALLENGED or BLOCK_UNCERTAIN from calibrate_belief.
        blocking_axiom: Axiom ID that fired (e.g., 'A3', 'A4').
        relation: The relation that was blocked (e.g., 'HARMS').
    """
    res = await asyncio.to_thread(
        hi_pai.escalate_block, object_id, verdict, blocking_axiom, relation
    )

    ruling = res.get("final_ruling", "FINAL_BLOCK")
    ruling_emoji = "🔴" if ruling == "FINAL_BLOCK" else "🟢"
    path = res.get("resolution_path", "UNKNOWN")
    conservative = res.get("conservative_default", False)
    log_lines = "\n  ".join(res.get("resolution_log", []))
    evidence_lines = "\n  • ".join(res.get("new_evidence", [])) or "None found"
    conservative_str = (
        "\n⚠️  CONSERVATIVE_DEFAULT: Classification unresolved. "
        "Submit new evidence via add_belief or ingest_observation."
        if conservative
        else ""
    )

    report = (
        f"<escalation_report axiom='{blocking_axiom}' "
        f"entity='{object_id}' path='{path}'>\n"
        f"Final Ruling: {ruling_emoji} {ruling}\n"
        f"Reasoning: {res.get('reasoning', '')}\n\n"
        f"Resolution Log:\n  {log_lines}\n\n"
        f"New Evidence:\n  • {evidence_lines}\n"
        f"{conservative_str}\n"
        f"</escalation_report>\n\n"
        f"Directive: {ruling} is the terminal routing decision. "
        f"If FINAL_BLOCK, the T1 constraint is structurally enforced. "
        f"If FINAL_PERMIT, the entity's classification was corrected by "
        f"evidence — action may proceed under T3 utility reasoning."
    )
    return report


@mcp.tool()
@mcp_tool_handler
async def verify_logic_foundation() -> str:
    """
    Triggers a machine-checked verification of the Paraclete T1 foundation.
    Runs Isabelle 2025-2 to prove consistency, gate soundness, and monotonicity.
    Requires Isabelle to be installed. This provides mathematical certainty
    that the core safety axioms are logically sound.
    """
    res = await asyncio.to_thread(hi_pai.world_model.paraclete.verify_foundation)

    status_emoji = "✅" if res["success"] else "❌"
    report = (
        f"### Logic Foundation Verification {status_emoji}\n\n"
        f"**Message**: {res['message']}\n\n"
        f"**Details**:\n```text\n{res['output'][-1000:] if res['output'] else 'No output available.'}\n```\n"
    )
    return report


def main():
    mcp.run()


@mcp.tool()
async def declare_class_hierarchy(parent_name: str, children_names: list[str]) -> str:
    """Dynamically declare a set of classes as subclasses of a parent in the ontology.
    Example: parent_name='Concept_Patient', children_names=['Concept_Elderly', 'Concept_Disabled']
    """
    res = await asyncio.to_thread(
        hi_pai.declare_class_hierarchy, parent_name, children_names
    )
    return f"Successfully declared hierarchy for {parent_name}: {', '.join(res)}"


@mcp.tool()
async def set_default_unclassified(parent_name: str) -> str:
    """Set the default ontology class for unclassified/low-confidence terms.
    Default is 'Concept_PossiblyPatient'."""
    return await asyncio.to_thread(hi_pai.set_default_unclassified, parent_name)


@mcp.tool()
async def list_protected_closure() -> str:
    """Return the list of all classes that fall under the protected 'Concept_Patient' hierarchy."""
    res = await asyncio.to_thread(hi_pai.list_protected_closure)
    return f"Protected Hierarchy Closure: {', '.join(res)}"


if __name__ == "__main__":
    main()

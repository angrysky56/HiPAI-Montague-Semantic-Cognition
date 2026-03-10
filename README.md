# HiPAI Montague Semantic Cognition

A prototype implementation blending Montague grammar for natural language semantics with a graph-based world model and Zettelkasten-inspired knowledge synthesis, built using FalkorDB.

## Overview

This project implements a novel semantic cognition pipeline:

1. **Parsing & Syntactic Analysis:** A custom shift-reduce parser (`hipai.core`) translates English surface forms into an abstract syntax tree representation, utilizing typed lambda calculus and Montague-style categorization.
2. **Semantic Evaluation:** The `hipai.semantics` module defines Semantic Types (Entities, Truth Values) and functional operations (Lambda Expressions, Application) to resolve meaning compositionally.
3. **World Model (Graph Database):** `hipai.world_model` integrates with FalkorDB to act as the agent's memory and knowledge base. Entities and propositions are stored as Nodes and Edges in the knowledge graph.
4. **Knowledge Synthesis:** Using a Zettelkasten-inspired method, `hipai.synthesis` processes low-level entity observations into abstract Concepts and Domains, creating a multi-tiered ontology autonomously.

## Requirements

- Python 3.12+
- `uv` (for dependency and environment management)
- FalkorDB (Running locally on `localhost:6379`)

### Setting up FalkorDB

You can easily run FalkorDB via Docker for local development:
```bash
docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
```

## Installation & Setup

We use `uv` for seamless project management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the test suite:
   ```bash
   uv run pytest tests/ -v
   ```

## Architecture

- `src/hipai/core.py`: Implements the `Lexicon` and `ShiftReduceParser` for syntactic categorization.
- `src/hipai/semantics.py`: Implements typed lambda calculus with `SemanticType`, `LambdaExpression`, and `evaluate()`.
- `src/hipai/world_model.py`: Handles connection to the local FalkorDB store, modeling entities and graph propositions.
- `src/hipai/synthesis.py`: Synthesizes graph structure notes (Concepts) from base entities based on shared properties.

## Example Usage

Create a script `example.py`:

```python
from hipai.core import Lexicon, ShiftReduceParser
from hipai.semantics import SemanticType, Entity, evaluate, create_lambda_for_property
from hipai.world_model import WorldModel
from hipai.synthesis import ZettelkastenSynthesizer

# 1. Initialize Parser
lex = Lexicon()
lex.add_word("Socrates", "NP")
lex.add_word("runs", "VI")
parser = ShiftReduceParser(lex)
tree = parser.parse("Socrates runs")

# 2. Assign Semantics
socrates = Entity("Socrates", repr_str="s")
runs_func = create_lambda_for_property("runs")

# 3. Evaluate Proposition
truth_val = evaluate(runs_func, socrates)
print(f"Proposition: {truth_val.repr_str}")

# 4. Integrate into World Model
wm = WorldModel(host="127.0.0.1", port=6379)
wm.create_entity("Socrates", properties={"prop_runs": truth_val.value})

# 5. Synthesize Concepts
synth = ZettelkastenSynthesizer(wm)
concepts = synth.synthesize_concepts()
print(f"Synthesized concepts: {concepts}")
```

Run with:
```bash
uv run python example.py
```

## MCP Context Integration

This project includes a Model Context Protocol (MCP) server that exposes the semantic world model to AI assistants like Claude, Cursor, or Windsurf.

To configure your AI assistant to use the HiPAI Montague Semantic Cognition tools, add the following to your MCP client configuration (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hipai-montague": {
      "command": "uv",
      "args": [
        "--directory",
        "/your-path-to/HiPAI-Montague-Semantic-Cognition",
        "run",
        "python",
        "-m",
        "hipai.mcp_server"
      ],
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

Once connected, your AI assistant will have access to the following capabilities:
- `add_belief`: Add a natural language fact or logic rule to the world model.
- `ingest_observation`: Ingest a cognitive observation in JSON format (entities and relations).
- `evaluate_hypothesis`: Test a statement against the current knowledge in the graph (entailment logic).
- `semantic_search`: Find nodes semantically related to a query using vector embeddings.
- `synthesize_concepts`: Discover abstract Structure Notes based on shared entity properties.
- `vector_synthesize_concepts`: Perform KMeans clustering to discover latent concepts in embedding space.
- `synthesize_domains`: Group related concepts into higher-level Main Structure Notes (Domains).
- `query_graph`: Execute raw OpenCypher queries against the knowledge graph.
- `get_current_state`: Retrieve a snapshot of the current nodes and edges.
- `clear_graph`: Reset the world model database.

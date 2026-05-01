# Technology Stack

## Core
- **Python**: >= 3.13 (Modern features, type hinting)
- **MCP Framework**: [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (Exposing tools and resources)

## Persistence
- **FalkorDB**: Primary graph database for deductive reasoning storage.
- **SQLite**: (Referenced by `hipai_world.db`) potentially used for local caching or lightweight state.

## AI & NLP
- **Sentence-Transformers**: Vector embeddings for semantic search.
- **Scikit-learn**: Clustering (KMeans) for concept synthesis.
- **Inflect**: Linguistic normalization (singular/plural).
- **BeautifulSoup4**: HTML/XML parsing for data ingestion.

## Infrastructure & Tooling
- **uv**: Package and environment management.
- **Ruff**: Fast linting and formatting.
- **Pytest**: Unit and integration testing.
- **Pydantic**: Robust data validation and settings management.
- **aiohttp**: Asynchronous HTTP client.

# Architecture

## Core Components

### 1. Semantic Engine (`semantics.py`)
Implements Montague semantics, including basic types (e, t) and complex functional types (<e,t>). Handles lambda expression composition and application.

### 2. World Model (`world_model.py`)
The persistence layer. Maps semantic entities and relations to a graph structure in FalkorDB. Supports semantic search (vector + graph) and constraint validation.

### 3. Synthesis Engine (`synthesis.py`)
The "brain" of the system. Contains `HIPAIManager` which orchestrates:
- **Belief Addition**: Parsing text into semantic observations.
- **Hypothesis Evaluation**: Testing if a hypothesis follows from the current world model.
- **Concept Synthesis**: Using clustering to discover higher-order domains and relations.

### 4. Data Models (`models.py`)
Pydantic-based definitions of the system's ontology: Individuals, Properties, Relations, Observations, and Axioms.

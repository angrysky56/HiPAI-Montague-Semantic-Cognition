"""
Semantic models for the HiPAI-Montague bridge.

This module defines the Pydantic models used to represent semantic entities,
properties, relations, and observations as they are mapped from natural
language to Montague-style formalisms.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class Individual(BaseModel):
    """
    Representation of an individual entity (Semantic Type: e).
    """

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the individual.",
    )
    name: str = Field(
        ..., description="Human-readable name or concept for the individual."
    )
    properties: list[str] = Field(
        default_factory=list,
        description=(
            "List of properties (type <e, t>) this individual possesses, "
            "e.g., 'Human', 'Mortal'."
        ),
    )


class Property(BaseModel):
    """
    Representation of a semantic property (Semantic Type: <e, t>).
    """

    name: str = Field(
        ..., description="The name of the property, e.g., 'Human', 'Mortal'."
    )
    description: str | None = Field(
        default=None, description="Optional description of the property."
    )


class Relation(BaseModel):
    """
    Representation of a binary relation (Semantic Type: <e, <e, t>>).
    """

    source_id: str = Field(..., description="ID of the subject/source individual.")
    target_id: str = Field(..., description="ID of the object/target individual.")
    relation_type: str = Field(
        ..., description="Type of the relation, e.g., 'Loves', 'Kills'."
    )


class TruthValue(BaseModel):
    """
    Representation of a truth value (Semantic Type: t).
    """

    value: bool = Field(..., description="The Boolean truth value.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this truth value.",
    )


class Observation(BaseModel):
    """
    An observation extracted from text representing a new proposition to be integrated.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text_source: str = Field(..., description="The original natural language sentence.")
    individuals: list[Individual] = Field(
        default_factory=list, description="Extracted individuals."
    )
    relations: list[Relation] = Field(
        default_factory=list, description="Extracted relations between individuals."
    )


class Contradiction(BaseModel):
    """
    Represents a logical contradiction found during reasoning.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str = Field(..., description="Description of the contradiction.")
    # The components that are in conflict. This is loosely structured for now.
    conflicting_assertions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The specific propositions or relations that contradict each other."
        ),
    )
    resolution_strategy: Literal["reject_new", "override_old", "escalate"] = Field(
        default="escalate", description="Strategy to resolve the contradiction."
    )

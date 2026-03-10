"""
Integration bridge between cognitive models and technical components.

This module provides the core orchestration logic for connecting high-level
semantic processing with underlying technical infrastructure.
"""

import json
from typing import Any

from .models import Observation


def get_observation_schema() -> dict[str, Any]:
    """
    Get the JSON Schema for the Observation model.
    This schema can be passed to an LLM to enforce structured output.
    """
    return Observation.model_json_schema()


def get_observation_schema_json(indent: int = 2) -> str:
    """
    Get the JSON Schema as a formatted string.
    """
    return json.dumps(get_observation_schema(), indent=indent)


def parse_observation(llm_output: str | dict[str, Any]) -> Observation:
    """
    Parse the LLM generated JSON into an Observation object.

    Args:
        llm_output: The JSON string or dictionary returned by the LLM.

    Returns:
        The validated Observation instance.
    """
    if isinstance(llm_output, str):
        return Observation.model_validate_json(llm_output)
    return Observation.model_validate(llm_output)

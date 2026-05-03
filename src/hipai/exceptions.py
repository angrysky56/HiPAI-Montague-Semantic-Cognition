"""
Custom exceptions for the HiPAI-Montague-Semantic-Cognition system.
"""

from typing import Any


class HiPAIError(Exception):
    """Base exception for all HiPAI errors."""

    pass


class AmbiguityDetectedError(HiPAIError):
    """
    Raised when a natural language input results in multiple valid logical interpretations.

    Attributes:
        possible_parses: A list of candidate interpretations (e.g., Observation objects or dictionaries).
    """

    def __init__(self, possible_parses: list[Any]):
        self.possible_parses = possible_parses
        super().__init__(
            f"Ambiguity detected: {len(possible_parses)} possible interpretations found."
        )

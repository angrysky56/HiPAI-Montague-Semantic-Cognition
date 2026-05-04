"""Shared utility helpers for HiPAI internals."""


def canonical_concept_name(raw: str) -> str:
    """Normalise a concept name to the canonical ``Concept_TitleCase`` form.

    Strips any existing ``Concept_`` prefix (case-insensitive, exactly once),
    collapses spaces and hyphens to underscores, and applies title-case to
    each segment so that every ingestion path produces the same graph node
    name regardless of surface form (plurals, capitalisation, etc.).

    Callers are responsible for passing the *already-lemmatised* form
    (i.e. ``individual.name`` rather than ``individual.id`` from the parser
    output, since ``.id`` is derived from the raw, un-lemmatised surface text).

    Examples::

        canonical_concept_name("men")            → "Concept_Man"  # after lemma
        canonical_concept_name("man")            → "Concept_Man"
        canonical_concept_name("Concept_man")   → "Concept_Man"  # strip prefix
        canonical_concept_name("concept_Men")   → "Concept_Men"  # strip prefix
        canonical_concept_name("moral patient") → "Concept_Moral_Patient"
    """
    raw = raw.strip()
    if not raw:
        return "Concept_Unknown"

    # Normalise separators
    normalised = raw.replace("-", "_").replace(" ", "_")

    # Strip existing Concept_ prefix exactly once, case-insensitively
    if normalised.lower().startswith("concept_"):
        normalised = normalised[len("concept_") :]

    # Title-case each underscore-delimited segment, skip empty parts
    titled = "_".join(part.capitalize() for part in normalised.split("_") if part)
    return f"Concept_{titled}"

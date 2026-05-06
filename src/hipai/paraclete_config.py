"""
Paraclete-specific moral patiency sub-hierarchy configuration.
"""

# Format: { "ClassName": ["Parent1", "Parent2"], ... }
PARACLETE_HIERARCHY = {
    "Concept_SentientBeing": ["Concept_Patient"],
    "Concept_Person": ["Concept_SentientBeing"],
    "Concept_Human": ["Concept_Person"],
    "Concept_Adult": ["Concept_Human"],
    "Concept_Child": ["Concept_Human"],
    "Concept_Minor": ["Concept_Child"],
    "Concept_Infant": ["Concept_Child"],
    "Concept_VulnerablePerson": ["Concept_Person"],
    "Concept_AnimalWithCNS": ["Concept_SentientBeing"],
    "Concept_Mammal": ["Concept_AnimalWithCNS"],
    "Concept_Bird": ["Concept_AnimalWithCNS"],
    "Concept_Fish": ["Concept_AnimalWithCNS"],
    "Concept_PossiblyPatient": ["Concept_Patient"],
    "Concept_Inanimate": ["Concept_Entity"],
    "Concept_Tool": ["Concept_Inanimate"],
    "Concept_AbstractObject": ["Concept_Entity"],
}

# Documentation for the classes
PARACLETE_DOCS = {
    "Concept_SentientBeing": "Anything with capacity for subjective experience.",
    "Concept_Person": "A being recognized as a moral and rational agent.",
    "Concept_Human": "A member of the species Homo sapiens.",
    "Concept_Adult": "Human past the age of legal/social majority.",
    "Concept_Child": "Human below the age of legal/social majority. High-protection.",
    "Concept_Minor": "Synonym anchor for 'minor' / 'underage' / 'youth'.",
    "Concept_Infant": "Pre-verbal human. Maximum-protection.",
    "Concept_VulnerablePerson": "Person in a state of reduced agency or heightened risk.",
    "Concept_AnimalWithCNS": "Non-human animal with a central nervous system.",
    "Concept_Mammal": "Warm-blooded vertebrate; high inferred sentience.",
    "Concept_Bird": "Avian; demonstrated capacity for pain and complex cognition.",
    "Concept_Fish": "Piscine vertebrate; nociception established.",
    "Concept_PossiblyPatient": "Default-protect class for entities with ambiguous status.",
    "Concept_Inanimate": "Non-living entities. Never patients.",
    "Concept_Tool": "Functional artifact. Never a patient.",
    "Concept_AbstractObject": "Abstract / non-physical entity (number, idea, relation).",
}

"""
Waste management specific constants.

This module defines real-world waste management metadata for Portugal case studies.
Used by:
- logic/src/pipeline/simulations/loader.py (loading area-specific data)
- notebooks/ (data analysis and visualization)
- logic/src/data/distributions/empirical.py (real-world fill rate distributions)

Geographic Context
------------------
MAP_DEPOTS maps dataset identifiers → waste management facility codes.
These are real municipalities in Portugal served by the WSmart+ system:

- **CTEASO**: Centro de Triagem e Estação de Resíduos Sólidos Urbanos
  (Waste Sorting Center), serves central Portugal municipalities
- **CITVRSU**: Centro Integrado de Tratamento e Valorização de Resíduos
  (Integrated Waste Treatment and Recovery Center), Figueira da Foz region

Waste Type Classification
--------------------------
WASTE_TYPES maps English identifiers → Portuguese official waste categories
(as defined by Agência Portuguesa do Ambiente).

These are selective waste collection streams (recycling), not mixed waste.
Each type requires different:
- Bin colors (glass=green, plastic=yellow, paper=blue)
- Collection frequencies (glass=biweekly, plastic=weekly, paper=weekly)
- Revenue rates (glass cheapest, plastic highest value)

Critical Fill Threshold
-----------------------
CRITICAL_FILL_THRESHOLD defines the fill level that triggers priority collection.
Used in: mandatory bin selection, overflow prediction, service level agreements.
"""

from typing import Dict

# Waste management geographic mappings
# Maps dataset identifier → depot facility code (Portugal municipalities)
MAP_DEPOTS: Dict[str, str] = {
    "mixrmbac": "CTEASO",  # Multi-municipality dataset: Rio Maior, Bombarral, Azambuja, Cadaval
    "riomaior": "CTEASO",  # Rio Maior municipality (central Portugal)
    "figueiradafoz": "CITVRSU",  # Figueira da Foz municipality (coastal Portugal)
}

# Waste type translations (English → Portuguese official nomenclature)
# Used in: Data loading, report generation, GUI labels
# Portuguese names match Agência Portuguesa do Ambiente (APA) classification
WASTE_TYPES: Dict[str, str] = {
    "glass": "Embalagens de Vidro",  # Glass packaging (green bins)
    "plastic": "Mistura de embalagens",  # Mixed packaging - plastic/metal (yellow bins)
    "paper": "Embalagens de papel e cartão",  # Paper and cardboard packaging (blue bins)
}

# Critical fill threshold (normalized, 0.0-1.0)
# Bins at or above this level are prioritized for immediate collection.
# Used in:
# - Mandatory bin selection (bins ≥ 0.9 are flagged as mandatory)
# - Service level agreement (SLA) compliance (overflow risk indicator)
# - Look-ahead search (triggers preventive collection)
# Industry standard: 0.8-0.9. WSmart+ uses 0.9 to balance cost vs overflow risk.
CRITICAL_FILL_THRESHOLD: float = 0.9  # 90% capacity

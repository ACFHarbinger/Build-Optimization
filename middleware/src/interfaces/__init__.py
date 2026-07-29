"""
Interfaces for WSmart-Route logic components.
DEFINING PROTOCOLS TO DECOUPLE MODULES.
"""

from .acceptance_criterion import IAcceptanceCriterion
from .bin_container import IBinContainer
from .context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)
from .env import IEnv
from .mandatory_selection import IMandatorySelectionStrategy
from .model import IModel
from .policy import IPolicy
from .route_constructor import IRouteConstructor
from .route_improvement import IRouteImprovement
from .tensor_dict_like import ITensorDictLike
from .traversable import ITraversable

__all__ = [
    "IAcceptanceCriterion",
    "IRouteConstructor",
    "IEnv",
    "IModel",
    "IMandatorySelectionStrategy",
    "IPolicy",
    "IRouteImprovement",
    "ITensorDictLike",
    "ITraversable",
    "IBinContainer",
    # Context / tracking types
    "SearchContext",
    "SearchPhase",
    "SelectionMetrics",
    "ConstructionMetrics",
    "AcceptanceMetrics",
    "ImprovementMetrics",
    "merge_context",
]

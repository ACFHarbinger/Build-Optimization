"""
Policies module for WSmart-Route.

Contains all neural and classical policy implementations for combinatorial
optimization. Use ``get_policy(name)`` to look up a policy class by its
short CLI name, or import the class directly.

Registries:
- ``POLICY_REGISTRY``: Maps short names (e.g. "am", "mdam") to policy classes.
- ``get_policy(name, **kwargs)``: Factory function for instantiation.
"""

import importlib

from torch import nn

from logic.src.models.common import (
    ConstructivePolicy,
    ImprovementPolicy,
    NonAutoregressivePolicy,
)
from logic.src.models.core.attention_model.deep_decoder_policy import DeepDecoderPolicy
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.core.attention_model.symnco_policy import SymNCOPolicy
from logic.src.models.core.dact.policy import DACTPolicy
from logic.src.models.core.deepaco.policy import DeepACOPolicy
from logic.src.models.core.gfacs.policy import GFACSPolicy
from logic.src.models.core.glop.policy import GLOPPolicy
from logic.src.models.core.hybrid_attention_model.hybrid_neural_heuristic_policy import NeuralHeuristicHybrid
from logic.src.models.core.hybrid_attention_model.hybrid_two_step_policy import HybridTwoStagePolicy
from logic.src.models.core.mdam.policy import MDAMPolicy
from logic.src.models.core.moe.policy import MoEPolicy
from logic.src.models.core.n2s.policy import N2SPolicy
from logic.src.models.core.nargnn.policy import NARGNNPolicy
from logic.src.models.core.neuopt.policy import NeuOptPolicy
from logic.src.models.core.pointer_network.policy import PointerNetworkPolicy
from logic.src.models.core.polynet.policy import PolyNetPolicy
from logic.src.models.core.temporal_attention_model.policy import TemporalAMPolicy

from .alns import VectorizedALNS
from .ant_colony_system import VectorizedACOPolicy
from .augmented_hybrid_volleyball_premier_league import VectorizedAHVPL
from .hgs import VectorizedHGS
from .hgs_alns import VectorizedHGSALNS
from .hybrid_volleyball_premier_league import VectorizedHVPL
from .iterated_local_search import IteratedLocalSearchPolicy

# Short-name registry: CLI model name -> (module_path, class_name)
# This enables ``get_policy("am")`` without importing everything eagerly.
_POLICY_REGISTRY_SPEC = {
    "am": ("logic.src.models.core.attention_model.policy", "AttentionModelPolicy"),
    "deep_decoder": ("logic.src.models.core.attention_model.deep_decoder_policy", "DeepDecoderPolicy"),
    "deepaco": ("logic.src.models.core.deepaco.policy", "DeepACOPolicy"),
    "gfacs": ("logic.src.models.core.gfacs.policy", "GFACSPolicy"),
    "glop": ("logic.src.models.core.glop.policy", "GLOPPolicy"),
    "mdam": ("logic.src.models.core.mdam.policy", "MDAMPolicy"),
    "moe": ("logic.src.models.core.moe.policy", "MoEPolicy"),
    "nargnn": ("logic.src.models.core.nargnn.policy", "NARGNNPolicy"),
    "n2s": ("logic.src.models.core.n2s.policy", "N2SPolicy"),
    "neuopt": ("logic.src.models.core.neuopt.policy", "NeuOptPolicy"),
    "dact": ("logic.src.models.core.dact.policy", "DACTPolicy"),
    "pointer": ("logic.src.models.core.pointer_network.policy", "PointerNetworkPolicy"),
    "polynet": ("logic.src.models.core.polynet.policy", "PolyNetPolicy"),
    "symnco": ("logic.src.models.core.attention_model.symnco_policy", "SymNCOPolicy"),
    "temporal": ("logic.src.models.core.temporal_attention_model.policy", "TemporalAMPolicy"),
    "hvpl": ("logic.src.models.policies.hybrid_volleyball_premier_league", "VectorizedHVPL"),
    "ahvpl": ("logic.src.models.policies.augmented_hybrid_volleyball_premier_league", "VectorizedAHVPL"),
    "hgs": ("logic.src.models.policies.hgs", "VectorizedHGS"),
    "hgs_alns": ("logic.src.models.policies.hgs_alns", "VectorizedHGSALNS"),
}


def get_policy_class(name: str) -> type:
    """
    Look up a policy class by its short CLI name.

    Args:
        name: Short policy name (e.g. "am", "mdam", "deepaco").

    Returns:
        The policy class (not instantiated).

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in _POLICY_REGISTRY_SPEC:
        raise ValueError(f"Unknown policy: {name!r}. Available: {sorted(_POLICY_REGISTRY_SPEC.keys())}")
    module_path, class_name = _POLICY_REGISTRY_SPEC[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_policy(name: str, **kwargs) -> nn.Module:
    """
    Create a policy instance by short CLI name.

    Args:
        name: Short policy name (e.g. "am", "mdam", "deepaco").
        **kwargs: Arguments passed to the policy constructor.

    Returns:
        Instantiated policy module.
    """
    cls = get_policy_class(name)
    return cls(**kwargs)


__all__ = [
    "ConstructivePolicy",
    "ImprovementPolicy",
    "AttentionModelPolicy",
    "DeepDecoderPolicy",
    "DeepACOPolicy",
    "GFACSPolicy",
    "GLOPPolicy",
    "MDAMPolicy",
    "NARGNNPolicy",
    "PolyNetPolicy",
    "TemporalAMPolicy",
    "PointerNetworkPolicy",
    "SymNCOPolicy",
    "MoEPolicy",
    "NeuOptPolicy",
    "DACTPolicy",
    "N2SPolicy",
    "NonAutoregressivePolicy",
    "get_policy_class",
    "get_policy",
    "VectorizedHGSALNS",
    "IteratedLocalSearchPolicy",
    "VectorizedACOPolicy",
    "VectorizedALNS",
    "VectorizedHGS",
    "VectorizedHVPL",
    "VectorizedAHVPL",
    "NeuralHeuristicHybrid",
    "HybridTwoStagePolicy",
]

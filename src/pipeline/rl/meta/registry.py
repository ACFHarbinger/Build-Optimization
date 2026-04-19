"""
Meta-Learning Strategy Registry.
"""

from typing import Dict, Type

from pipeline.rl.meta.contextual_bandits import WeightContextualBandit
from pipeline.rl.meta.hypernet_strategy import HyperNetworkStrategy
from pipeline.rl.meta.multi_objective.weight_optimizer import MORLWeightOptimizer
from pipeline.rl.meta.td_learning import CostWeightManager
from pipeline.rl.meta.weight_optimizer import RewardWeightOptimizer
from pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy

META_STRATEGY_REGISTRY: Dict[str, Type[WeightAdjustmentStrategy]] = {
    "rnn": RewardWeightOptimizer,
    "rwa": RewardWeightOptimizer,
    "bandit": WeightContextualBandit,
    "morl": MORLWeightOptimizer,
    "tdl": CostWeightManager,
    "hypernet": HyperNetworkStrategy,
}


def get_meta_strategy(name: str, **kwargs) -> WeightAdjustmentStrategy:
    """Get meta-learning strategy by name."""
    strategy_cls = META_STRATEGY_REGISTRY.get(name.lower())
    if strategy_cls is None:
        raise ValueError(f"Unknown meta strategy: {name}")
    return strategy_cls(**kwargs)

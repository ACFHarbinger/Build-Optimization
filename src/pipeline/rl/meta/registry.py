"""
Meta-Learning Strategy Registry.
"""

from typing import Dict, Type

from logic.src.pipeline.rl.meta.contextual_bandits import WeightContextualBandit
from logic.src.pipeline.rl.meta.hypernet_strategy import HyperNetworkStrategy
from logic.src.pipeline.rl.meta.multi_objective.weight_optimizer import MORLWeightOptimizer
from logic.src.pipeline.rl.meta.td_learning import CostWeightManager
from logic.src.pipeline.rl.meta.weight_optimizer import RewardWeightOptimizer
from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy

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

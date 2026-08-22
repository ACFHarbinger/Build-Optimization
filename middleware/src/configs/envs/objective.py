"""Objective / reward-weight configuration.

Ported from the sibling WSmart-Route ``configs.envs.objective`` module so
``from configs import Config`` can finish importing (``ModelConfig.reward``
and ``NeuralAgentConfig.reward`` both depend on this type). Build-Optimization
does not currently consume these VRP-shaped weights at runtime; they exist so
the inherited RL-pipeline config graph type-checks and imports.
"""

from dataclasses import dataclass


@dataclass
class ObjectiveConfig:
    """Configuration for problem objectives and reward weights.

    Attributes:
        cost_weight: Weight for length/distance in the cost function.
        waste_weight: Weight for waste collection in the cost function.
        overflow_penalty: Penalty factor for overflows.
    """

    cost_weight: float = 1.0
    waste_weight: float = 1.0
    overflow_penalty: float = 1.0

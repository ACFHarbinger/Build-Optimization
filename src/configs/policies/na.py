"""
Neural policy configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from logic.src.configs.envs.objective import ObjectiveConfig
from logic.src.configs.models.model import ModelConfig
from logic.src.configs.policies.other.mandatory_selection import MandatorySelectionConfig
from logic.src.configs.policies.other.route_improvement import RouteImprovingConfig


@dataclass
class NeuralAgentConfig:
    """Configuration for Neural Agent policy.

    Attributes:
        model: Model configuration (architecture and load path).
        decoding: Decoding strategy configuration.
        reward: Objective/reward weights configuration.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is a VRP with Profits.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
    vrpp: bool = True

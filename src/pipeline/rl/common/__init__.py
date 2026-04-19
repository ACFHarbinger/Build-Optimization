"""
Training common subpackage for WSmart-Route.

This package contains training utilities and common components
like epoch preparation, dataset regeneration, and training hooks.
"""

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.common.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    MeanBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    SharedBaseline,
    WarmupBaseline,
    get_baseline,
)
from logic.src.pipeline.rl.common.reward_scaler import RewardScaler
from logic.src.pipeline.rl.common.reward_scaler_batch import BatchRewardScaler
from logic.src.pipeline.rl.common.route_improvement import (
    EfficiencyOptimizer,
    calculate_efficiency,
    decode_routes,
)
from logic.src.pipeline.rl.common.trainer import WSTrainer

from . import base as base
from . import baselines as baselines
from . import reward_scaler as reward_scaler
from . import reward_scaler_batch as reward_scaler_batch
from . import route_improvement as route_improvement
from . import trainer as trainer

__all__ = [
    "RL4COLitModule",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "WarmupBaseline",
    "POMOBaseline",
    "MeanBaseline",
    "SharedBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "WSTrainer",
    "RewardScaler",
    "BatchRewardScaler",
    "EfficiencyOptimizer",
    "decode_routes",
    "calculate_efficiency",
]

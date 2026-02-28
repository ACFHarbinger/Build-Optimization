"""
Training common subpackage for WSmart-Route.

This package contains training utilities and common components
like epoch preparation, dataset regeneration, and training hooks.
"""

from pipeline.rl.common.base import LitModule
from pipeline.rl.common.baselines import (
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
from pipeline.rl.common.reward_scaler import RewardScaler
from pipeline.rl.common.reward_scaler_batch import BatchRewardScaler
from pipeline.rl.common.trainer import WSTrainer

__all__ = [
    "LitModule",
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
]

"""
RL Pipeline module for WSmart-Route.
"""

from pipeline.rl.common.base import RL4COLitModule
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
from pipeline.rl.core.a2c import A2C
from pipeline.rl.core.adaptive_imitation import AdaptiveImitation
from pipeline.rl.core.dr_grpo import DRGRPO
from pipeline.rl.core.gdpo import GDPO
from pipeline.rl.core.gspo import GSPO
from pipeline.rl.core.imitation import ImitationLearning
from pipeline.rl.core.mvmoe_am import MVMoE_AM
from pipeline.rl.core.mvmoe_pomo import MVMoE_POMO
from pipeline.rl.core.pomo import POMO
from pipeline.rl.core.ppo import PPO
from pipeline.rl.core.reinforce import REINFORCE
from pipeline.rl.core.sapo import SAPO
from pipeline.rl.core.symnco import SymNCO
from pipeline.rl.meta.hrl import HRLModule
from pipeline.rl.meta.module import MetaRLModule

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
    "REINFORCE",
    "PPO",
    "A2C",
    "SAPO",
    "GSPO",
    "GDPO",
    "AdaptiveImitation",
    "DRGRPO",
    "ImitationLearning",
    "MetaRLModule",
    "HRLModule",
    "POMO",
    "SymNCO",
    "MVMoE_POMO",
    "MVMoE_AM",
]

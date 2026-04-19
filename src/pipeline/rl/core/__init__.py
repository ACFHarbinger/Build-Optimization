"""
RL Pipeline module for WSmart-Route.
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
from logic.src.pipeline.rl.core.a2c import A2C
from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation
from logic.src.pipeline.rl.core.dr_grpo import DRGRPO
from logic.src.pipeline.rl.core.gdpo import GDPO
from logic.src.pipeline.rl.core.gspo import GSPO
from logic.src.pipeline.rl.core.imitation import ImitationLearning
from logic.src.pipeline.rl.core.mvmoe_am import MVMoE_AM
from logic.src.pipeline.rl.core.mvmoe_pomo import MVMoE_POMO
from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.pipeline.rl.core.ppo import PPO
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.pipeline.rl.core.sapo import SAPO
from logic.src.pipeline.rl.core.symnco import SymNCO
from logic.src.pipeline.rl.meta.hrl import HRLModule
from logic.src.pipeline.rl.meta.module import MetaRLModule

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

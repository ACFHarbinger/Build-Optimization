"""
RL Baselines sub-package.
"""

from __future__ import annotations

from typing import Optional

from torch import nn

from .base import Baseline
from .critic import CriticBaseline
from .exponential import ExponentialBaseline
from .mean import MeanBaseline
from .none import NoBaseline
from .pomo import POMOBaseline
from .rollout import RolloutBaseline
from .shared_critic import SharedBaseline
from .warmup import WarmupBaseline

# Baseline registry
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
    "warmup": WarmupBaseline,
    "pomo": POMOBaseline,
    "mean": MeanBaseline,
    "shared": SharedBaseline,
}


def get_baseline(name: str, policy: Optional[nn.Module] = None, **kwargs) -> Baseline:
    """Get baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")

    kwargs.pop("policy", None)  # Avoid duplicate or string policy from hparams
    baseline = BASELINE_REGISTRY[name](**kwargs)
    if policy is not None:
        baseline.setup(policy)
    return baseline


__all__ = [
    "Baseline",
    "MeanBaseline",
    "NoBaseline",
    "CriticBaseline",
    "SharedBaseline",
    "ExponentialBaseline",
    "POMOBaseline",
    "RolloutBaseline",
    "WarmupBaseline",
    "BASELINE_REGISTRY",
    "get_baseline",
]

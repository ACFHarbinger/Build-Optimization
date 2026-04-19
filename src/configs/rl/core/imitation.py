"""Imitation specific configuration."""

from dataclasses import dataclass
from typing import Any, Union

from logic.src.configs.rl.policies import (
    ACOConfig,
    ALNSConfig,
    HGSALNSConfig,
    HGSConfig,
    ILSConfig,
    RLSConfig,
)

# Type alias for expert policy configurations
ExpertPolicyConfig = Union[HGSConfig, ALNSConfig, ILSConfig, RLSConfig, ACOConfig, HGSALNSConfig]


@dataclass
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Configuration for the expert policy to imitate.
                      Can be any of: HGSConfig, ALNSConfig, ILSConfig, RLSConfig, ACOConfig, HGSALNSConfig.
        loss_fn: Loss function to use ('nll' for negative log-likelihood, 'mse' for mean squared error).
    """

    policy_config: Any = None
    loss_fn: str = "nll"

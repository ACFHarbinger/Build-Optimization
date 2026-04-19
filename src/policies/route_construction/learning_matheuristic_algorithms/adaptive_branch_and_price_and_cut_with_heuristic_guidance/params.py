"""
Configuration parameters for ABPC-HG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logic.src.configs.policies.abpc_hg import ABPCHGConfig


@dataclass
class ABPCHGParams:
    """
    Configuration parameters for Adaptive Branch-and-Price-and-Cut with Heuristic Guidance.
    """

    gamma: float = 0.95
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: ABPCHGConfig) -> ABPCHGParams:
        """Create ABPCHGParams from an ABPCHGConfig dataclass.

        Args:
            config: ABPCHGConfig dataclass with solver parameters.

        Returns:
            ABPCHGParams instance with values from config.
        """
        return cls(
            gamma=getattr(config, "gamma", 0.95),
            seed=getattr(config, "seed", None),
        )

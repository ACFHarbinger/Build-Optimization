"""
GFACS Policy: GFlowNet Ant Colony System.

Combines GFACSEncoder (heatmap prediction) with ACODecoder for construction.
Adds learnable partition function (logZ) for Trajectory Balance loss training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.core.deepaco.policy import DeepACOPolicy
from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.encoders.gfacs.encoder import GFACSEncoder


class GFACSPolicy(DeepACOPolicy):
    """
    GFACS Policy.

    Extends DeepACOPolicy with a learnable logZ parameter for GFlowNet training
    (Trajectory Balance Loss).
    """

    def __init__(
        self,
        encoder: Optional[GFACSEncoder] = None,
        decoder: Optional[ACODecoder] = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        env_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize GFACSPolicy.

        Args:
            encoder: GFACSEncoder instance.
            decoder: ACODecoder instance.
            embed_dim: Embedding dimension.
            num_encoder_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            n_ants: Number of ants in ACO.
            n_iterations: Number of ACO iterations.
            alpha: Pheromone importance.
            beta: Heuristic importance.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to use local search in ACO.
            env_name: Environment name.
            **kwargs: Additional arguments.
        """
        if encoder is None:
            encoder = GFACSEncoder(
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                **kwargs,
            )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_local_search=use_local_search,
            env_name=env_name,
            **kwargs,
        )

        # Learnable log partition function for Trajectory Balance loss
        # Initialize to 0.0 or a small random value
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for GFACS.

        Returns:
            Dict containing:
                - reward: Rewards for constructed solutions.
                - log_likelihood: Log probabilities of solutions.
                - actions: Constructed solutions.
                - heatmap: Predicted edge heatmap.
                - logZ: Learnable partition function value (scalar).
        """
        # GFACS requires all ants for Trajectory Balance loss
        if "return_all" not in kwargs:
            kwargs["return_all"] = True

        out = super().forward(td, env, num_starts=num_starts, **kwargs)
        # Add logZ to output for loss calculation (e.g. in Trajectory Balance Loss)
        out["logZ"] = self.logZ
        if "ls_actions" in out:
            out["ls_logZ"] = self.logZ
        return out

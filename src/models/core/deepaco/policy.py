"""
DeepACO Policy: Combines encoder and decoder for end-to-end inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.encoder import (
    NonAutoregressiveEncoder,
)
from logic.src.models.common.non_autoregressive.policy import (
    NonAutoregressivePolicy,
)
from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.encoders.deepaco.encoder import DeepACOEncoder


class DeepACOPolicy(NonAutoregressivePolicy):
    """
    DeepACO Policy.

    Combines DeepACOEncoder (heatmap prediction) with ACODecoder
    (ant colony solution construction) for end-to-end inference.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
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
        Initialize DeepACOPolicy.

        Args:
            encoder: DeepACOEncoder instance.
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
            encoder = DeepACOEncoder(
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                **kwargs,
            )
        if decoder is None:
            decoder = ACODecoder(
                n_ants=n_ants,
                n_iterations=n_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                use_local_search=use_local_search,
            )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward.

        Args:
            td (TensorDict): Description of td.
            env (RL4COEnvBase): Description of env.
            num_starts (int): Description of num_starts.
            kwargs (Any): Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        # Encode: predict heatmap
        encoder = cast(NonAutoregressiveEncoder, self.encoder)
        heatmap = encoder(td, **kwargs)

        # Decode: construct solution(s) from heatmap using ACO
        decoder = cast(ACODecoder, self.decoder)
        out = decoder.construct(td, heatmap, env, num_starts=num_starts, **kwargs)

        out["heatmap"] = heatmap
        return out

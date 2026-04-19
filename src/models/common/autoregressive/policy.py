"""autoregressive_policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import autoregressive_policy
"""

from typing import Any, Dict, Optional

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase

from .constructive import ConstructivePolicy
from .decoder import AutoregressiveDecoder
from .encoder import AutoregressiveEncoder


class AutoregressivePolicy(ConstructivePolicy):
    """
    Base class for autoregressive policies.

    Combines an AR encoder with an AR decoder to form a complete policy.
    Inherits from ConstructivePolicy to leverage standardized decoding strategies.
    """

    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize AutoregressivePolicy."""
        # Use placeholders for base initialization if not provided
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            seed=seed,
            device=device,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode + decode.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions.
            strategy: Decoding strategy.
            num_starts: Number of solution starts.

        Returns:
            Dictionary containing reward, log_likelihood, and actions.
        """
        # Encode
        embeddings = self.encoder(td, **kwargs) if self.encoder is not None else None

        # Decode
        if self.decoder is not None:
            # Note: Many decoders in WSmart-Route implement their own loop.
            # We assume the decoder handles the AR process.
            log_p, actions = self.decoder(td, embeddings, env, strategy=strategy, num_starts=num_starts, **kwargs)
        else:
            raise ValueError("AutoregressivePolicy requires a decoder.")

        # Calculate reward
        reward = env.get_reward(td, actions)

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "actions": actions,
        }

"""improvement_decoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import improvement_decoder
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class ImprovementDecoder(nn.Module, ABC):
    """
    Base class for improvement decoders.

    Improvement decoders predict moves or operators to apply to the current solution.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize ImprovementDecoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict improvement moves.

        Args:
            td: TensorDict containing current state.
            embeddings: Encoded state and solution.
            env: Environment for state transitions.

        Returns:
            Tuple of (log_likelihood, actions/moves).
        """
        raise NotImplementedError

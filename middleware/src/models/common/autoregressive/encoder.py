"""autoregressive_encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import autoregressive_encoder
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class AutoregressiveEncoder(nn.Module, ABC):
    """
    Base class for autoregressive encoders.

    AR encoders take a problem instance and produce initial embeddings
    that represent the problem state.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize AutoregressiveEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Compute initial embeddings for the problem instance.

        Args:
            td: TensorDict containing problem instance.

        Returns:
            Embeddings tensor or tuple of tensors representing the encoded state.
        """
        raise NotImplementedError

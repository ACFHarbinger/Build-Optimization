"""improvement_encoder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import improvement_encoder
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class ImprovementEncoder(nn.Module, ABC):
    """
    Base class for improvement encoders.

    Improvement encoders take both the problem instance and the current solution
    to produce embeddings representing the current state of search.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize ImprovementEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Compute embeddings for the problem instance and current solution.

        Args:
            td: TensorDict containing problem instance and 'tour' or current solution.

        Returns:
            Embeddings tensor or tuple of tensors.
        """
        raise NotImplementedError

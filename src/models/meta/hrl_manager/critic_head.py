"""critic_head.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import critic_head
"""

import torch
from torch import nn


class CriticHead(nn.Module):
    """
    Critic head for value estimation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize Class.

        Args:
            input_dim (int): Description of input_dim.
            hidden_dim (int): Description of hidden_dim.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Description of x.

        Returns:
            Any: Description of return value.
        """
        return self.net(x)

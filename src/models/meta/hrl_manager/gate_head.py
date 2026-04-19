"""gate_head.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import gate_head
"""

import torch
from torch import nn


class GateHead(nn.Module):
    """
    Head for routing decision (gate).
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
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Description of x.

        Returns:
            Any: Description of return value.
        """
        return self.net(x)

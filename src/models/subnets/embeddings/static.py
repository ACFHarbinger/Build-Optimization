"""static.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import static
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn


class StaticEmbedding(nn.Module):
    """Static embedding: No dynamic updates."""

    def __init__(self, *args, **kwargs):
        """Initialize StaticEmbedding."""
        super().__init__()

    def forward(self, td: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero embeddings."""
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

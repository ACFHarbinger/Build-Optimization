"""reward_scaler_batch.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import reward_scaler_batch
"""

import torch


class BatchRewardScaler:
    """
    Per-batch reward scaler (no running statistics).

    Normalizes each batch independently.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize BatchRewardScaler.

        Args:
            eps: Numerical stability constant.
        """
        self.eps = eps

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores within batch using population statistics."""
        mean = scores.mean()
        std = scores.std(correction=0)
        # Handle constant or near-constant batches to avoid numerical instability
        if std < self.eps:
            return torch.zeros_like(scores)
        return (scores - mean) / std

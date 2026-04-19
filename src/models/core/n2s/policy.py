"""policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import policy
"""

from __future__ import annotations

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import N2SDecoder
from .encoder import N2SEncoder


class N2SPolicy(ImprovementPolicy):
    """
    N2S Policy: Iterative neighborhood search.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs,
    ):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            num_heads (int): Description of num_heads.
            k_neighbors (int): Description of k_neighbors.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = N2SEncoder(embed_dim, num_heads, k_neighbors)
        self.decoder = N2SDecoder(embed_dim)

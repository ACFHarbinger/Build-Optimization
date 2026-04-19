"""policy.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import policy
"""

from __future__ import annotations

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import NeuOptDecoder
from .encoder import NeuOptEncoder


class NeuOptPolicy(ImprovementPolicy):
    """
    NeuOpt Policy: Iterative guided search.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs,
    ):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            num_heads (int): Description of num_heads.
            num_layers (int): Description of num_layers.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = NeuOptEncoder(embed_dim, num_heads, num_layers)
        self.decoder = NeuOptDecoder(embed_dim)

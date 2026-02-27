"""
Encoder Config module.
"""

from dataclasses import dataclass, field
from typing import Optional

from .activation_function import ActivationConfig
from .normalization import NormalizationConfig


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""

    type: str = "gat"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    n_sublayers: Optional[int] = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    dropout: float = 0.1
    mask_inner: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"

    # Aggregation params
    aggregation_graph: str = "avg"
    aggregation_node: str = "sum"

    # Other
    spatial_bias_scale: float = 1.0
    hyper_expansion: int = 4

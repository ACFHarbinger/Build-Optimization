"""
Decoder Config module.
"""

from dataclasses import dataclass, field
from typing import Optional

from .activation_function import ActivationConfig
from .decoding import DecodingConfig
from .normalization import NormalizationConfig


@dataclass
class DecoderConfig:
    """Decoder architecture and decoding configuration."""

    type: str = "attention"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    dropout: float = 0.1
    mask_logits: bool = True
    connection_type: str = "residual"
    n_predictor_layers: Optional[int] = None
    tanh_clipping: float = 10.0
    hyper_expansion: int = 4

"""
Model Config module.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..envs.objective import ObjectiveConfig
from .decoder import DecoderConfig
from .encoder import EncoderConfig


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Name of the model architecture (e.g., 'am', 'deep_decoder').
        encoder: Encoder configuration.
        decoder: Decoder configuration.
        temporal_horizon: Temporal horizon for the model.
        policy_config: Policy configuration string.
    """

    name: str = "am"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    temporal_horizon: int = 0
    policy_config: Optional[str] = None
    load_path: Optional[str] = None

"""Attention model component factory."""

from __future__ import annotations

from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class AttentionComponentFactory(NeuralComponentFactory):
    """Factory for Angle-based Attention Models."""

    def create_encoder(
        self,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create Graph Attention Encoder."""
        return GraphAttentionEncoder(
            norm_config=norm_config,
            activation_config=activation_config,
            **kwargs,
        )

    def create_decoder(
        self,
        decoder_type: str = "attention",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(
            decoder_type,
            norm_config=norm_config,
            activation_config=activation_config,
            **kwargs,
        )

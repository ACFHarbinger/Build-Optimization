"""GCN component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class GCNComponentFactory(NeuralComponentFactory):
    """Factory for Graph Convolutional Network Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create GCN Encoder."""
        return GraphConvolutionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)

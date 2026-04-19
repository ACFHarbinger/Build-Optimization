"""TGC component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.tgc import TransGraphConvEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class TGCComponentFactory(NeuralComponentFactory):
    """Factory for Transformer Graph Convolution Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create Transformer Graph Convolution Encoder."""
        return TransGraphConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)

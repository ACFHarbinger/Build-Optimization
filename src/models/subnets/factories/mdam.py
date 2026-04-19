"""MDAM component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.mdam import MDAMGraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class MDAMComponentFactory(NeuralComponentFactory):
    """Factory for MDAM Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create MDAM Graph Attention Encoder."""
        return MDAMGraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "mdam", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)

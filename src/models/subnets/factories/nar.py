"""NAR component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class NARComponentFactory(NeuralComponentFactory):
    """Factory for Non-Autoregressive Models (DeepACO, GFACS, NARGNN)."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create NARGNN Encoder."""
        return NARGNNEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)

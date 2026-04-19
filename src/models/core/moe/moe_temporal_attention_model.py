"""
Mixture of Experts (MoE) Temporal Attention Model.
"""

from logic.src.models.core.temporal_attention_model import TemporalAttentionModel
from logic.src.models.subnets.factories import MoEComponentFactory


class MoETemporalAttentionModel(TemporalAttentionModel):
    """
    Temporal Attention Model with Mixture of Experts (MoE) Encoder.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        n_encode_sublayers=None,
        n_decode_layers=None,
        dropout_rate=0.1,
        normalization="batch",
        n_heads=8,
        num_experts=4,
        k=2,
        noisy_gating=True,
        **kwargs,
    ):
        """
        Initialize the MoE Temporal Attention Model.
        """
        # Create the MoE Factory with specific params
        component_factory = MoEComponentFactory(num_experts=num_experts, k=k, noisy_gating=noisy_gating)

        super(MoETemporalAttentionModel, self).__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            component_factory=component_factory,  # Inject custom factory
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            normalization=normalization,
            n_heads=n_heads,
            **kwargs,
        )

    def embed_and_transform(self, input, edges=None):
        """
        Embed and transform input.
        Wraps parent method to handle potential 4D output from MoE encoder.
        """
        embeddings, _ = super().embed_and_transform(input, edges)  # type: ignore[misc]

        if embeddings.dim() == 4:
            # [Batch, Graph, K/Heads, Dim] -> [Batch, Graph, Dim]
            # Collapse the extra dimension (likely experts or heads)
            embeddings = embeddings.mean(dim=2)

        return embeddings, None

    @property
    def total_experts(self):
        """Get the total number of experts in the model."""
        return self.encoder.num_experts if hasattr(self.encoder, "num_experts") else 0

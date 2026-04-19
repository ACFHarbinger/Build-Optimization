"""
Multi-Head Attention (MHA) logic for constructuve decoders.
"""

import math
from typing import Optional

import torch


def one_to_many_logits(
    query: torch.Tensor,
    glimpse_K: torch.Tensor,
    glimpse_V: torch.Tensor,
    logit_K: torch.Tensor,
    mask: torch.Tensor,
    n_heads: int,
    graph_mask: Optional[torch.Tensor] = None,
    dist_bias: Optional[torch.Tensor] = None,
    mask_val: float = -math.inf,
    tanh_clipping: float = 10.0,
) -> torch.Tensor:
    """
    Compute attention logits for a query against multiple nodes.
    """
    batch_size, num_steps, embed_dim = query.size()
    key_size = embed_dim // n_heads
    # Compute heads: [batch_size, n_heads, num_steps, key_size]
    q_heads = query.view(batch_size, num_steps, n_heads, key_size).transpose(1, 2)

    # glimpse_K: [batch_size, n_heads, graph_size, key_size]
    # glimpse_V: [batch_size, n_heads, graph_size, key_size]
    # logit_K: [batch_size, n_heads, graph_size, key_size]

    # Attention scores: [batch_size, n_heads, num_steps, graph_size]
    attn_scores = torch.matmul(q_heads, glimpse_K.transpose(-2, -1)) / math.sqrt(key_size)

    # Masking
    if mask is not None:
        # mask: [batch_size, num_steps, graph_size]
        mask_expanded = mask.unsqueeze(1)  # [batch_size, 1, num_steps, graph_size]
        attn_scores = attn_scores.masked_fill(mask_expanded, mask_val)

    if graph_mask is not None:
        attn_scores = attn_scores.masked_fill(graph_mask.unsqueeze(1).unsqueeze(2), mask_val)

    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Glimpse: [batch_size, n_heads, num_steps, key_size]
    glimpse = torch.matmul(attn_probs, glimpse_V)

    # Project back
    glimpse_combined = glimpse.transpose(1, 2).contiguous().view(batch_size, num_steps, embed_dim)

    # Final logits
    # query_final: [batch_size, n_heads, num_steps, key_size]
    # We use a separate projection for the final logit K as in AM
    q_final = glimpse_combined.view(batch_size, num_steps, n_heads, key_size).transpose(1, 2)
    logits = torch.matmul(q_final, logit_K.transpose(-2, -1)) / math.sqrt(key_size)

    # Combine heads by averaging? No, usually AM uses MHA and then one final projection.
    # Actually, the implementation in GlimpseDecoder averages heads for logits.
    logits = logits.mean(dim=1)

    if dist_bias is not None:
        logits = logits + dist_bias

    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    if mask is not None:
        logits = logits.masked_fill(mask, mask_val)

    return logits


def make_heads(v: torch.Tensor, n_heads: int, num_steps: Optional[int] = None) -> torch.Tensor:
    """Reshape tensor into heads."""
    batch_size, graph_size, embed_dim = v.size()
    if num_steps is None:
        num_steps = graph_size

    # heads: [batch_size, n_heads, graph_size, key_size]
    return v.view(batch_size, graph_size, n_heads, embed_dim // n_heads).transpose(1, 2)

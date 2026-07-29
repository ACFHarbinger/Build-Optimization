"""
Attention computation for MDAM.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_mdam_logits(
    query: torch.Tensor,
    glimpse_K: torch.Tensor,
    glimpse_V: torch.Tensor,
    logit_K: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    project_out: torch.nn.Module,
    tanh_clipping: float,
    mask_inner: bool,
    mask_logits: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention-based logits."""
    batch_size, num_steps, embed_dim = query.size()
    key_size = embed_dim // num_heads

    # Reshape query for multi-head
    glimpse_Q = query.view(batch_size, num_steps, num_heads, 1, key_size).permute(2, 0, 1, 3, 4)

    # Compute compatibility
    compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(key_size)

    if mask_inner and mask is not None:
        compatibility_mask = ~mask[None, :, None, None, :].expand_as(compatibility)
        compatibility[compatibility_mask] = -math.inf

    # Compute attention and aggregate
    heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

    # Project out
    glimpse = project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, num_heads * key_size))

    # Compute final logits
    logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

    # Apply tanh clipping
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    # Apply mask
    if mask_logits and mask is not None:
        logits[~mask[:, None, :]] = -math.inf

    # Convert to log probs
    logprobs = F.log_softmax(logits, dim=-1)

    return logprobs, glimpse.squeeze(-2)

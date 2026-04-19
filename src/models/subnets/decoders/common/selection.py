"""
Node selection utilities for autoregressive decoders.

This module provides unified action selection logic used across multiple
decoder implementations (GlimpseDecoder, PointerDecoder, PolyNetDecoder, MDAMDecoder).

Consolidates duplicated greedy/sampling selection code from:
- logic/src/models/subnets/decoders/glimpse/decoder.py (_select_node)
- logic/src/models/subnets/decoders/ptr/decoder.py (decode)
- logic/src/models/subnets/decoders/polynet/decoder.py (forward)
- logic/src/models/subnets/decoders/mdam/cache.py (_decode_probs)
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.constants.models import NUMERICAL_EPSILON


def select_action(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strategy: str = "greedy",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Select action from probability distribution using greedy or sampling strategy.

    This function is used in autoregressive decoders to select the next node
    to visit based on the attention-derived probability distribution over nodes.

    Parameters
    ----------
    probs : torch.Tensor
        Probability distribution over actions/nodes.
        Shape: (batch_size, num_nodes) or (batch_size, num_actions)
        Should be a valid probability distribution (non-negative, sums to ~1.0).
    mask : Optional[torch.Tensor], default=None
        Boolean mask indicating valid actions (True = valid, False = invalid).
        Shape: (batch_size, num_nodes) or broadcastable to probs shape.
        If provided, invalid actions are masked out and probabilities are renormalized.
    strategy : str, default="greedy"
        Selection strategy:
        - "greedy": Select action with highest probability (argmax)
        - "sampling": Sample action from probability distribution (multinomial)
    generator : Optional[torch.Generator], default=None
        Generator for sampling. If provided, used for sampling.

    Returns
    -------
    torch.Tensor
        Selected action indices.
        Shape: (batch_size,)
        Values are integers in range [0, num_nodes-1].

    Examples
    --------
    Greedy selection:
    >>> probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])  # (B=2, N=3)
    >>> selected = select_action(probs, strategy="greedy")
    >>> print(selected)  # tensor([1, 2]) - argmax per batch

    Sampling with mask:
    >>> probs = torch.tensor([[0.5, 0.3, 0.2]])  # (B=1, N=3)
    >>> mask = torch.tensor([[True, False, True]])  # Node 1 is invalid
    >>> selected = select_action(probs, mask=mask, strategy="sampling")
    >>> print(selected)  # tensor([0] or [2]) - samples only from valid actions

    Greedy with mask:
    >>> probs = torch.tensor([[0.1, 0.7, 0.2]])  # (B=1, N=3)
    >>> mask = torch.tensor([[True, False, True]])  # Node 1 is invalid
    >>> selected = select_action(probs, mask=mask, strategy="greedy")
    >>> print(selected)  # tensor([2]) - highest probability among valid actions

    Notes
    -----
    - When mask is provided, probabilities are:
      1. Zeroed out for invalid actions (mask == False)
      2. Renormalized to sum to 1.0 over valid actions
    - For greedy selection, ties are broken arbitrarily (first occurrence)
    - For sampling, small epsilon (1e-8) is added to prevent sampling failures
    - This function assumes probs are already non-negative (e.g., from softmax/exp)

    See Also
    --------
    - Used in GlimpseDecoder._select_node()
    - Used in PointerDecoder.decode()
    - Used in PolyNetDecoder.forward()
    - Used in MDAMDecoder (via mdam/cache.py _decode_probs)
    """
    # Apply mask if provided
    if mask is not None:
        # Zero out invalid actions
        probs = probs.masked_fill(~mask, 0.0)
        # Renormalize to valid probability distribution
        probs = probs / (probs.sum(dim=-1, keepdim=True) + NUMERICAL_EPSILON)

    # Select action based on strategy
    if strategy == "greedy":
        # Greedy: Select action with highest probability
        return probs.argmax(dim=-1)
    else:
        # Sampling: Sample from probability distribution
        # Clamp to prevent numerical issues in multinomial
        probs = probs.clamp(min=NUMERICAL_EPSILON)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def select_action_log_prob(
    log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strategy: str = "greedy",
) -> torch.Tensor:
    """
    Select action from log-probability distribution.

    Convenience wrapper around select_action() that accepts log-probabilities
    instead of probabilities. Useful when working with log-softmax outputs.

    Parameters
    ----------
    log_probs : torch.Tensor
        Log-probability distribution over actions.
        Shape: (batch_size, num_nodes)
    mask : Optional[torch.Tensor], default=None
        Boolean mask indicating valid actions.
    strategy : str, default="greedy"
        Selection strategy: "greedy" or "sampling".

    Returns
    -------
    torch.Tensor
        Selected action indices.
        Shape: (batch_size,)

    Examples
    --------
    >>> log_probs = torch.log_softmax(logits, dim=-1)  # (B, N)
    >>> selected = select_action_log_prob(log_probs, strategy="greedy")

    Notes
    -----
    This function converts log-probabilities to probabilities (exp) before
    calling select_action(). For numerical stability, log-probabilities
    should be computed using log_softmax, not log(softmax).
    """
    # Convert log-probabilities to probabilities
    probs = log_probs.exp()
    return select_action(probs, mask=mask, strategy=strategy)

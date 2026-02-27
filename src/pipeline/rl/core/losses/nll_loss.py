"""Negative Log-Likelihood loss for behavior cloning."""

from __future__ import annotations

from torch import Tensor


def nll_loss(
    log_likelihood: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute Negative Log-Likelihood loss for behavior cloning.

    This is the standard loss for imitation learning where we maximize
    the log probability of expert actions under our policy.

    Args:
        log_likelihood: Log probabilities of actions, shape (batch_size,) or (batch_size, seq_len).
        reduction: Reduction method ('mean', 'sum', 'none').

    Returns:
        Scalar loss tensor (or per-sample if reduction='none').

    Example:
        >>> log_ll = policy(td, env, actions=expert_actions)["log_likelihood"]
        >>> loss = nll_loss(log_ll)
    """
    nll = -log_likelihood

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

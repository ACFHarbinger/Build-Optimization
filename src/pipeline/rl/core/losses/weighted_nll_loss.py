"""Weighted Negative Log-Likelihood loss for advantage-weighted imitation."""

from __future__ import annotations

from torch import Tensor


def weighted_nll_loss(
    log_likelihood: Tensor,
    weights: Tensor,
    reduction: str = "mean",
    normalize_weights: bool = False,
) -> Tensor:
    """
    Compute Weighted Negative Log-Likelihood loss.

    This loss weights the NLL by an advantage or reward difference,
    allowing the model to focus more on actions where the expert
    significantly outperforms the current policy.

    Args:
        log_likelihood: Log probabilities of actions, shape (batch_size,).
        weights: Per-sample weights (e.g., expert_reward - agent_reward), shape (batch_size,).
        reduction: Reduction method ('mean', 'sum', 'none').
        normalize_weights: If True, normalize weights to have mean 1.0.

    Returns:
        Scalar loss tensor (or per-sample if reduction='none').

    Example:
        >>> log_ll = policy(td, env, actions=expert_actions)["log_likelihood"]
        >>> advantage = expert_reward - agent_reward
        >>> loss = weighted_nll_loss(log_ll, advantage)
    """
    if normalize_weights and weights.numel() > 1:
        weights = weights / (weights.mean().abs() + 1e-8)

    weighted_nll = -log_likelihood * weights

    if reduction == "mean":
        return weighted_nll.mean()
    elif reduction == "sum":
        return weighted_nll.sum()
    elif reduction == "none":
        return weighted_nll
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

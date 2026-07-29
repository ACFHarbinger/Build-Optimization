"""Jensen-Shannon Divergence loss for policy distillation."""

from __future__ import annotations

from torch import Tensor


def js_divergence_loss(
    log_probs: Tensor,
    target_log_probs: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute Jensen-Shannon Divergence between two distributions.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    JS divergence is symmetric, always finite, and its square root
    is a metric. It is often more stable than KL divergence.

    Args:
        log_probs: Log probabilities from the policy, shape (batch_size, num_actions).
        target_log_probs: Log probabilities from the target, shape (batch_size, num_actions).
        reduction: Reduction method ('mean', 'sum', 'none', 'batchmean').

    Returns:
        Scalar loss tensor.
    """
    # Convert log probs to probs
    P = log_probs.exp()
    Q = target_log_probs.exp()

    # Average distribution
    M = 0.5 * (P + Q)
    log_M = M.log()

    # KL(P || M)
    kl_p_m = (P * (log_probs - log_M)).sum(dim=-1)

    # KL(Q || M)
    kl_q_m = (Q * (target_log_probs - log_M)).sum(dim=-1)

    # JS divergence
    js = 0.5 * (kl_p_m + kl_q_m)

    if reduction == "mean":
        return js.mean()
    elif reduction == "sum":
        return js.sum()
    elif reduction == "batchmean":
        return js.sum() / js.size(0)
    elif reduction == "none":
        return js
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', 'batchmean', or 'none'.")

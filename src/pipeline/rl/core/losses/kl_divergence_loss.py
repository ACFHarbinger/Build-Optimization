"""KL Divergence loss for policy distillation."""

from __future__ import annotations

from torch import Tensor


def kl_divergence_loss(
    log_probs: Tensor,
    target_log_probs: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute KL Divergence between policy and target distribution.

    KL(target || policy) = sum(target * log(target / policy))
                         = sum(target * (log_target - log_policy))

    This is useful for policy distillation where we want to match
    the full action distribution of an expert, not just the argmax action.

    Args:
        log_probs: Log probabilities from the policy, shape (batch_size, num_actions).
        target_log_probs: Log probabilities from the target, shape (batch_size, num_actions).
        reduction: Reduction method ('mean', 'sum', 'none', 'batchmean').

    Returns:
        Scalar loss tensor (or per-sample if reduction='none').

    Example:
        >>> kl = kl_divergence_loss(policy_logits.log_softmax(-1), expert_logits.log_softmax(-1))
    """
    # Convert log probs to probs for the target
    target_probs = target_log_probs.exp()

    # KL divergence: sum over action dimension
    kl = (target_probs * (target_log_probs - log_probs)).sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "batchmean":
        return kl.sum() / kl.size(0)
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', 'batchmean', or 'none'.")


def reverse_kl_divergence_loss(
    log_probs: Tensor,
    target_log_probs: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute Reverse KL Divergence (mode-seeking).

    KL(policy || target) = sum(policy * log(policy / target))

    Reverse KL is mode-seeking and useful when we want the policy
    to concentrate on the modes of the target distribution.

    Args:
        log_probs: Log probabilities from the policy, shape (batch_size, num_actions).
        target_log_probs: Log probabilities from the target, shape (batch_size, num_actions).
        reduction: Reduction method ('mean', 'sum', 'none', 'batchmean').

    Returns:
        Scalar loss tensor.
    """
    probs = log_probs.exp()
    kl = (probs * (log_probs - target_log_probs)).sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "batchmean":
        return kl.sum() / kl.size(0)
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Invalid reduction: {reduction}.")

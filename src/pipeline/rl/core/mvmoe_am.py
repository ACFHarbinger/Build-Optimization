"""
MVMoE (Multi-View Mixture of Experts) models.

Thin wrappers around POMO and REINFORCE that inject MoE-specific kwargs
into the AttentionModelPolicy, enabling expert-specialized encoding and
decoding for improved routing performance.

Reference:
    Zhou, J., Wu, Y., Song, Q., Ma, S., & Cao, J. (2024).
    MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts.
    arXiv:2405.01029.
"""

from __future__ import annotations

from typing import Optional

from torch import nn

from logic.src.constants.models import DEFAULT_MOE_KWARGS
from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.pipeline.rl.core.reinforce import REINFORCE


class MVMoE_AM(REINFORCE):
    """
    MVMoE with Attention Model backbone.

    Combines MoE-enhanced AttentionModelPolicy with standard REINFORCE
    and a rollout baseline.

    Reference:
        Zhou et al. (2024). MVMoE: Multi-Task Vehicle Routing Solver
        with Mixture-of-Experts. arXiv:2405.01029.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        moe_kwargs: Optional[dict] = None,
        baseline: str = "rollout",
        **kwargs,
    ):
        """
        Initialize MVMoE_AM.

        Args:
            policy: Pre-built policy. If None, creates an AttentionModelPolicy
                    with MoE kwargs.
            moe_kwargs: MoE configuration for encoder and decoder.
            baseline: Baseline type (default "rollout").
            **kwargs: Passed to REINFORCE/RL4COLitModule.
        """
        if moe_kwargs is None:
            moe_kwargs = DEFAULT_MOE_KWARGS

        if policy is None:
            env_name = kwargs.get("env_name", "vrpp")
            policy = AttentionModelPolicy(
                env_name=env_name,
                moe_kwargs=moe_kwargs,
                **{k: v for k, v in kwargs.pop("policy_kwargs", {}).items()},
            )

        super().__init__(
            policy=policy,
            baseline=baseline,
            **kwargs,
        )

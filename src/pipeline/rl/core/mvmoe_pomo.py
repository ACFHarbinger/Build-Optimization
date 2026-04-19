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

from typing import Callable, Optional, Union

from torch import nn

from logic.src.constants.models import DEFAULT_MOE_KWARGS
from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.pipeline.rl.core.pomo import POMO


class MVMoE_POMO(POMO):
    """
    MVMoE with POMO backbone.

    Combines MoE-enhanced AttentionModelPolicy with POMO's multi-start
    decoding and shared baseline for improved routing performance.

    Reference:
        Zhou et al. (2024). MVMoE: Multi-Task Vehicle Routing Solver
        with Mixture-of-Experts. arXiv:2405.01029.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        moe_kwargs: Optional[dict] = None,
        num_augment: int = 8,
        augment_fn: Union[str, Callable] = "dihedral8",
        first_aug_identity: bool = True,
        num_starts: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize MVMoE_POMO.

        Args:
            policy: Pre-built policy. If None, creates an AttentionModelPolicy
                    with MoE kwargs.
            moe_kwargs: MoE configuration for encoder and decoder. Uses defaults
                        (4 experts, top-2, noisy gating) if None.
            num_augment: Number of data augmentations (default 8 for dihedral).
            augment_fn: Augmentation function.
            first_aug_identity: Whether first augmentation is identity.
            num_starts: Number of multi-start nodes.
            **kwargs: Passed to POMO (and REINFORCE/RL4COLitModule).
        """
        if moe_kwargs is None:
            moe_kwargs = DEFAULT_MOE_KWARGS

        if policy is None:
            env_name = kwargs.get("env_name", "vrpp")
            # Inject MoE kwargs into policy, using MVMoE recommended defaults
            policy = AttentionModelPolicy(
                env_name=env_name,
                n_encode_layers=kwargs.pop("num_encoder_layers", 6),
                normalization=kwargs.pop("normalization", "instance"),
                moe_kwargs=moe_kwargs,
                **{k: v for k, v in kwargs.pop("policy_kwargs", {}).items()},
            )

        super().__init__(
            num_augment=num_augment,
            augment_fn=augment_fn,
            first_aug_identity=first_aug_identity,
            num_starts=num_starts,
            policy=policy,
            **kwargs,
        )

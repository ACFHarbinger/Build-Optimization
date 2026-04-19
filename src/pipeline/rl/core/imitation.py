"""
Imitation Learning algorithm implementation.
Trains a policy to mimic an expert solver (e.g., HGS, Local Search).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from tensordict import TensorDict

from logic.src.configs.rl.policies import (
    ACOConfig,
    ALNSConfig,
    HGSALNSConfig,
    HGSConfig,
    ILSConfig,
    RLSConfig,
)
from logic.src.models.policies.alns import VectorizedALNS
from logic.src.models.policies.ant_colony_system import VectorizedACOPolicy
from logic.src.models.policies.hgs import VectorizedHGS
from logic.src.models.policies.hgs_alns import VectorizedHGSALNS
from logic.src.models.policies.iterated_local_search import IteratedLocalSearchPolicy
from logic.src.models.policies.random_local_search import RandomLocalSearchPolicy
from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.core.losses import (
    js_divergence_loss,
    kl_divergence_loss,
    nll_loss,
    reverse_kl_divergence_loss,
    weighted_nll_loss,
)
from logic.src.utils.data.rl_utils import safe_td_copy


class ImitationLearning(RL4COLitModule):
    """
    Imitation Learning / Supervised Learning.

    Minimizes the difference between the policy's predicted action probabilities
    and the actions taken by an expert solver (e.g., HGS, Local Search).

    Features:
    - On-the-fly expert data generation or pre-computed dataset support.
    - Configurable Loss (NLL, Weighted NLL, KL, JS).
    - Support for multiple expert types.
    """

    def __init__(
        self,
        policy_config: Any,  # Expert policy configuration (HGSConfig, ALNSConfig, etc.)
        env_name: str,
        loss_fn: str = "nll",
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize ImitationLearning module.

        Args:
            policy_config: Expert policy configuration object (HGSConfig, ALNSConfig, etc.).
            env_name: Environment name for the expert policy.
            loss_fn: Name of loss function to use ('nll', 'kl', 'js', etc.).
            **kwargs: Arguments passed to RL4COLitModule.
        """
        # Baseline is not used in IL, but we keep the structure
        kwargs["baseline"] = "none"

        # Exclude non-serializable objects from hyperparameters
        self.save_hyperparameters(ignore=["policy_config", "env", "policy", "kwargs", "generator"])

        super().__init__(**kwargs)

        # Create expert policy from config
        self.expert_policy = self._create_expert_policy(policy_config, env_name, seed, device)
        self.expert_name = type(policy_config).__name__.replace("Config", "").lower()

        # Map loss functions
        self._loss_map = {
            "nll": nll_loss,
            "weighted_nll": weighted_nll_loss,
            "kl": kl_divergence_loss,
            "reverse_kl": reverse_kl_divergence_loss,
            "js": js_divergence_loss,
        }
        self.loss_fn_name = loss_fn
        self.loss_fn = self._loss_map.get(loss_fn, nll_loss)

    def _create_expert_policy(self, policy_config: Any, env_name: str, seed: int, device: str) -> Any:
        """Create expert policy from configuration.

        Args:
            policy_config: Expert policy configuration (HGSConfig, ALNSConfig, etc.).
            env_name: Environment name for the policy.
            seed: Random seed for reproducibility.
            device: Device to run the policy on.

        Returns:
            Initialized expert policy instance.
        """
        # Map config types to policy classes
        config_to_policy_map = {
            HGSConfig: VectorizedHGS,
            ALNSConfig: VectorizedALNS,
            HGSALNSConfig: VectorizedHGSALNS,
            RLSConfig: RandomLocalSearchPolicy,
            ILSConfig: IteratedLocalSearchPolicy,
            ACOConfig: VectorizedACOPolicy,
        }

        # Get the policy class for this config type
        policy_cls = config_to_policy_map.get(type(policy_config))
        if policy_cls is None:
            raise ValueError(
                f"Unknown policy config type: {type(policy_config)}. "
                f"Supported types: {list(config_to_policy_map.keys())}"
            )

        # Convert config to dict and add env_name
        config_dict = asdict(policy_config)
        config_dict["env_name"] = env_name
        config_dict["device"] = device
        config_dict["seed"] = seed

        # Create and return the policy
        return policy_cls(**config_dict)

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Compute Imitation Learning loss.

        Args:
            td: Current environment state.
            out: Policy output dictionary (not used for expert selection).
            batch_idx: Batch index.
            env: Environment instance.

        Returns:
            Scalar loss tensor.
        """
        # 1. Generate Expert Solutions
        if self.expert_policy is None:
            raise ValueError("expert_policy must be provided for ImitationLearning.")

        with torch.no_grad():
            expert_out = self.expert_policy(td, self.env)
            expert_actions = expert_out["actions"]

        # 2. Clone state and reset to ensure clean state for teacher forcing
        td_clone = safe_td_copy(td)
        td_clone = self.env.reset(td_clone)

        # 3. Teacher Forcing: Get policy log probabilities for expert actions
        teacher_out = self.policy(td_clone, self.env, actions=expert_actions)
        log_likelihood = teacher_out["log_likelihood"]

        # 4. Compute Loss using injected loss function
        if self.loss_fn_name == "weighted_nll":
            expert_reward = expert_out.get("reward", torch.ones_like(log_likelihood))
            loss = weighted_nll_loss(log_likelihood, weights=expert_reward, reduction="mean")
        elif self.loss_fn_name in ["kl", "reverse_kl", "js"]:
            # Distillation requires target distribution (logits or log_probs)
            target_log_probs = expert_out.get("log_probs", None)
            if target_log_probs is None:
                print(
                    f"[ImitationLearning] Warning: {self.loss_fn_name} requires target log_probs. Falling back to NLL."
                )
                loss = nll_loss(log_likelihood, reduction="mean")
            else:
                loss = self._loss_map[self.loss_fn_name](log_likelihood, target_log_probs, reduction="mean")  # type: ignore[operator]
        else:
            loss = nll_loss(log_likelihood, reduction="mean")

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/log_likelihood", log_likelihood.mean())
        if hasattr(expert_out, "get") or isinstance(expert_out, dict):
            er = expert_out.get("reward", None)
            if er is not None:
                self.log("train/expert_reward", er.mean())
            ec = expert_out.get("cost", None)
            if ec is not None:
                self.log("train/expert_cost", ec.mean())

        return loss

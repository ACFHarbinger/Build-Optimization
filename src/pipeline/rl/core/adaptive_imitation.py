"""
Adaptive Imitation Learning + RL algorithm.
Combines PPO/REINFORCE with Imitation Learning using an annealing schedule.
"""

from __future__ import annotations

import json
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
from logic.src.pipeline.rl.core.losses import (
    kl_divergence_loss,
    nll_loss,
    reverse_kl_divergence_loss,
    weighted_nll_loss,
)
from logic.src.pipeline.rl.core.reinforce import REINFORCE


class AdaptiveImitation(REINFORCE):
    """
    Combined RL + Imitation Learning with Adaptive Weight Scheduling.

    Loss = Loss_RL + lambda * Loss_IL

    Schedule:
    - lambda starts at `il_weight`
    - decays by `il_decay` each epoch
    - resets to `il_weight` if validation reward dominates patience (simulated annealing reheating)
    """

    def __init__(
        self,
        policy_config: Any,
        env_name: str,
        il_weight: float = 1.0,
        il_decay: float = 0.95,
        patience: int = 5,
        threshold: float = 0.05,
        decay_step: int = 1,
        epsilon: float = 1e-5,
        loss_fn: str = "weighted_nll",
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize AdaptiveImitation module.

        Args:
            policy_config: Expert policy configuration object (HGSConfig, ALNSConfig, etc.).
            env_name: Environment name for the expert policy.
            il_weight: Initial weight for imitation loss.
            il_decay: Decay factor for IL weight each epoch.
            patience: Epochs without improvement before resetting IL weight.
            threshold: Threshold for reannealing.
            decay_step: Number of epochs to decay IL weight.
            epsilon: Epsilon for improvement detection.
            loss_fn: Name of loss function to use ('weighted_nll', 'nll', etc.).
            seed: Random seed for reproducibility.
            **kwargs: Arguments passed to REINFORCE.
        """
        # Exclude non-serializable objects from hyperparameters
        self.save_hyperparameters(ignore=["policy_config", "env", "policy", "generator"])
        if hasattr(self, "hparams"):
            keys_to_remove = []
            allowed_types = (int, float, str, bool, type(None), torch.Tensor)
            for k, v in self.hparams.items():
                if k in ["expert_policy", "env", "policy", "policy_config"]:
                    keys_to_remove.append(k)
                    continue
                if isinstance(v, allowed_types):
                    continue
                if isinstance(v, (dict, list, tuple)):
                    try:
                        json.dumps(v)
                        continue
                    except (TypeError, OverflowError):
                        pass
                keys_to_remove.append(k)
            for k in keys_to_remove:
                self.hparams.pop(k, None)
        super().__init__(**kwargs)

        # Create expert policy from config
        self.expert_policy = self._create_expert_policy(policy_config, env_name, seed, device)
        self.il_weight = il_weight
        self.initial_il_weight = il_weight
        self.il_decay = il_decay
        self.patience = patience
        self.threshold = threshold
        self.decay_step = decay_step
        self.epsilon = epsilon
        self.stop_il = False

        # Map loss functions
        self._loss_map = {
            "nll": nll_loss,
            "weighted_nll": weighted_nll_loss,
            "kl": kl_divergence_loss,
            "reverse_kl": reverse_kl_divergence_loss,
        }
        self.loss_fn_name = loss_fn
        self.loss_fn = self._loss_map.get(loss_fn, weighted_nll_loss)

        # State tracking
        self.wait = 0
        self.best_reward = float("-inf")
        self.current_il_weight = il_weight

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

    def _sanitize_td(self, td: TensorDict) -> TensorDict:
        """
        Sanitize TensorDict to remove potential recursive references or unnecessary nesting.

        This constructs a new TensorDict containing only the leaf Tensors from the original,
        breaking any reference cycles that might cause RecursionError during cloning or printing.
        """
        # Iterate over all keys in the TensorDict and keep only Tensors
        # to ensure we break any nesting or recursive structures.
        safe_data = {}
        for key in td.keys():
            val = td.get(key)
            if isinstance(val, torch.Tensor):
                safe_data[key] = val

        return TensorDict(safe_data, batch_size=td.batch_size, device=td.device)

    def calculate_loss(  # noqa: C901
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,  # Accept env to match RL4COLitModule.shared_step
    ) -> torch.Tensor:
        """
        Compute Combined Loss: RL + IL.
        """
        # 1. RL Loss (REINFORCE)
        # Note: This computes advantage, baseline, etc.
        rl_loss = super().calculate_loss(td, out, batch_idx, env=env)

        # 2. Imitation Loss (Cross Entropy)
        # Generate expert actions
        if not self.stop_il:
            with torch.no_grad():
                expert_out = self.expert_policy(td, self.env)
                expert_actions = expert_out["actions"]
                expert_reward = expert_out.get("reward", None)
        else:
            expert_reward = None
            expert_actions = None
            expert_reward = None
            expert_cost = None
            expert_out = None

        # Conditional Imitation: Check if expert provides significant improvement
        do_imitation = not self.stop_il
        current_reward = out.get("reward")

        # Robust comparison for expert_reward and current_reward
        if do_imitation and expert_reward is not None and current_reward is not None:
            # Handle cases where they might be MagicMocks in tests
            try:
                e_max = expert_reward.max() if hasattr(expert_reward, "max") else expert_reward
                c_min = current_reward.min() if hasattr(current_reward, "min") else current_reward

                # Only compare if both are comparable (not mocks that return mocks)
                if (
                    isinstance(e_max, (torch.Tensor, float, int))
                    and isinstance(c_min, (torch.Tensor, float, int))
                    and e_max <= c_min + self.threshold
                ):
                    do_imitation = False
                    print("\n[AdaptiveImitation] Stopping IL for this batch due to lack of improvement")
            except Exception:
                pass

        if do_imitation:
            # If we need reward-based weighting, ensure rewards are available
            if self.loss_fn_name == "weighted_nll" and (expert_reward is None or current_reward is None):
                # print("[AdaptiveImitation] Falling back to non-weighted IL because rewards are missing")
                pass

            td_il = self.env.reset(td)

            # Use safety get for policy output
            il_out = self.policy(td_il, self.env, actions=expert_actions)
            log_likelihood = il_out.get("log_likelihood")

            if log_likelihood is None:
                return rl_loss

            # Compute modular loss
            if self.loss_fn_name == "weighted_nll" and expert_reward is not None and current_reward is not None:
                il_loss = weighted_nll_loss(
                    log_likelihood,
                    weights=(expert_reward - current_reward),
                    reduction="mean",
                )
            elif self.loss_fn_name == "nll":
                il_loss = nll_loss(log_likelihood, reduction="mean")
            elif self.loss_fn_name in ["kl", "reverse_kl", "js"] and expert_out is not None:
                target_log_probs = expert_out.get("log_probs", None)
                if target_log_probs is None:
                    # Fallback
                    if expert_reward is not None and current_reward is not None:
                        il_loss = weighted_nll_loss(log_likelihood, (expert_reward - current_reward), reduction="mean")
                    else:
                        il_loss = nll_loss(log_likelihood, reduction="mean")
                else:
                    il_loss = self._loss_map[self.loss_fn_name](log_likelihood, target_log_probs, reduction="mean")  # type: ignore[operator]
            else:
                # Default fallback
                if expert_reward is not None and current_reward is not None:
                    il_loss = weighted_nll_loss(log_likelihood, (expert_reward - current_reward), reduction="mean")
                else:
                    il_loss = nll_loss(log_likelihood, reduction="mean")
        else:
            il_loss = torch.tensor(0.0, device=self.device)

        # Combined Loss
        total_loss = rl_loss + self.current_il_weight * il_loss

        # Logging
        self.log("train/rl_loss", rl_loss, on_step=False, on_epoch=True)
        self.log("train/il_loss", il_loss, on_step=False, on_epoch=True)
        self.log("train/il_weight", self.current_il_weight, on_step=False, on_epoch=True)

        if expert_reward is not None:
            self.log("train/expert_reward", expert_reward.mean(), on_step=False, on_epoch=True)
        expert_cost = expert_out.get("cost", None) if expert_out is not None else None
        if expert_cost is not None:
            self.log("train/expert_cost", expert_cost.mean(), on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        """
        Update IL weight based on validation performance.

        Decays IL weight each epoch, or resets if patience is exceeded.
        """
        super().on_train_epoch_end()

        current_reward = self.trainer.callback_metrics.get("val/reward")
        if self.current_il_weight <= self.threshold:
            self.stop_il = True
        if current_reward is not None and not self.stop_il:
            # Check for improvement
            if current_reward > self.best_reward + self.epsilon:
                self.best_reward = current_reward.item()
                self.wait = 0
            else:
                self.wait += 1

            # Schedule Logic
            if self.wait >= self.patience:
                # Re-heat / Reset
                self.current_il_weight = self.initial_il_weight
                self.wait = 0
                print(f"\n[AdaptiveImitation] Re-heating IL weight to {self.initial_il_weight}")
            else:
                # Decay
                self.current_il_weight *= self.il_decay
                print(f"\n[AdaptiveImitation] IL weight decayed to {self.current_il_weight}")

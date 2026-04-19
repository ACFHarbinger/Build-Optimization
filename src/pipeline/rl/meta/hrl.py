"""
Hierarchical Reinforcement Learning (HRL) module.

Implements a Manager-Worker architecture:
- Manager: MandatoryManager (decides mandatory bins and dispatch)
- Worker: ConstructivePolicy (decides the route)

The manager outputs:
- mandatory_action: Which bins must be collected (boolean mask)
- gate_action: Whether to dispatch vehicles
"""

from __future__ import annotations

from typing import cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.constructive import ConstructivePolicy
from logic.src.models.meta.hrl_manager import MandatoryManager


class HRLModule(pl.LightningModule):
    """
    Lightning module for Hierarchical RL.

    Coordinates a high-level manager and a low-level worker.
    """

    def __init__(
        self,
        manager: MandatoryManager,
        worker: ConstructivePolicy,
        env: RL4COEnvBase,
        lr: float = 1e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        lambda_mask_aux: float = 0.0,
        entropy_coef: float = 0.1,
        **kwargs,
    ):
        """
        Initialize HRLModule.

        Args:
            manager: High-level MandatoryManager for mandatory selection and dispatch.
            worker: Low-level ConstructivePolicy for routing.
            env: RL environment.
            lr: Learning rate for manager optimizer.
            gamma: Discount factor for returns.
            clip_eps: PPO clipping parameter.
            ppo_epochs: Number of PPO optimization epochs.
            lambda_mask_aux: Weight for auxiliary mandatory loss (BCE with target).
            entropy_coef: Entropy regularization coefficient.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["manager", "worker", "env", "kwargs", "generator"])
        self.manager = manager
        self.worker = worker
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.lambda_mask_aux = lambda_mask_aux
        self.entropy_coef = entropy_coef

        # Disable automatic optimization for PPO multiple updates
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """
        Combined training step for Manager and Worker with PPO.
        """
        opt = cast(torch.optim.Optimizer, self.optimizers())

        # --- 1. Collection Phase ---
        td = self.env.reset(batch)

        # Prepare inputs for manager
        static = td["locs"]
        if "waste_history" in td.keys():
            dynamic = td["waste_history"]
        elif "fill_history" in td.keys():
            dynamic = td["fill_history"]
        else:
            dynamic = td["waste"].unsqueeze(-1).expand(-1, -1, 10)

        global_features = torch.stack(
            [
                td["waste"].mean(dim=1),
                td["waste"].max(dim=1)[0],
            ],
            dim=-1,
        )

        # Manager Decision (Stores states/actions in manager memory)
        # mandatory_action: (B, N) where 1 = must collect, 0 = optional
        mandatory_action, gate_action, manager_value = self.manager.select_action(
            static, dynamic, global_features, deterministic=False
        )

        # Worker Decision (Routing)
        dispatch_indices = (gate_action == 1).nonzero().squeeze(-1)
        total_reward = torch.zeros(td.batch_size, device=td.device)

        if len(dispatch_indices) > 0:
            td_worker = td[dispatch_indices]
            # Apply manager's mandatory selection to the TensorDict
            # mandatory = True means bin must be collected
            mandatory_mask = mandatory_action[dispatch_indices].bool()
            # Depot (index 0) should never be mandatory
            mandatory_mask[:, 0] = False
            td_worker["mandatory"] = mandatory_mask
            out_worker = self.worker(td_worker, self.env)
            total_reward[dispatch_indices] = out_worker["reward"]

        # Store rewards in manager buffer (assuming 1-step horizon for now)
        self.manager.rewards.append(total_reward)

        # Log metrics
        self.log("train/manager_gate_rate", gate_action.float().mean())
        self.log("train/manager_mandatory_rate", mandatory_action.float().mean())
        self.log("train/reward", total_reward.mean(), prog_bar=True)

        # --- 2. PPO Optimization Phase ---
        # Replicates logic from manager_train.py::update_manager_ppo

        # Prepare Tensors from Buffer
        # For now we handle single-step (T=1), but structure allows accumulation if we loops
        T = len(self.manager.rewards)
        B = self.manager.rewards[0].size(0)

        rewards_tensor = torch.stack(self.manager.rewards)  # (T, B)
        # Compute Returns (G_t)
        returns_tensor = torch.zeros_like(rewards_tensor)
        running_return = torch.zeros(B, device=self.device)
        for t in reversed(range(T)):
            running_return = rewards_tensor[t] + self.gamma * running_return
            returns_tensor[t] = running_return

        returns = returns_tensor.flatten()  # (T*B)

        # Concatenate memory buffers
        # Assuming select_action populated these
        old_states_static = torch.cat(self.manager.states_static)
        old_states_dynamic = torch.cat(self.manager.states_dynamic)
        old_states_global = torch.cat(self.manager.states_global)
        old_mandatory_actions = torch.cat(self.manager.actions_mandatory)
        old_gate_actions = torch.cat(self.manager.actions_gate)
        old_log_probs_mandatory = torch.cat(self.manager.log_probs_mandatory)
        old_log_probs_gate = torch.cat(self.manager.log_probs_gate)
        old_values = torch.cat(self.manager.values).squeeze(-1)
        old_target_mandatory = torch.cat(self.manager.target_mandatory) if self.manager.target_mandatory else None

        # Calculate Advantages
        advantages = returns - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # PPO Inner Loops
        old_states_static.size(0)
        mean_loss = 0

        # HICRA Credit Assignment weights
        # Higher weight for instances with critical nodes (overflow > 0.9)
        b_overflow = (old_states_dynamic[:, :, -1] > 0.9).float().sum(dim=1)
        credit_weight = 1.0 + (b_overflow * 0.5)
        credit_weight = credit_weight / (credit_weight.mean() + 1e-8)

        for _ in range(int(self.ppo_epochs)):
            # Forward pass for current policy
            mandatory_logits, gate_logits, values = self.manager(
                old_states_static, old_states_dynamic, old_states_global
            )
            values = values.squeeze(-1)

            # New Log Probs
            mandatory_dist = torch.distributions.Categorical(logits=mandatory_logits)
            gate_dist = torch.distributions.Categorical(logits=gate_logits)

            new_log_probs_mandatory = mandatory_dist.log_prob(old_mandatory_actions).sum(dim=1)
            new_log_probs_gate = gate_dist.log_prob(old_gate_actions)

            new_log_probs = new_log_probs_mandatory + new_log_probs_gate
            old_log_probs = old_log_probs_mandatory + old_log_probs_gate

            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate Loss with Credit Assignment
            surr1 = ratio * advantages * credit_weight
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages * credit_weight
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy = mandatory_dist.entropy().mean() + gate_dist.entropy().mean()

            loss = actor_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # Aux Loss (if target mandatory available - supervised signal)
            if self.lambda_mask_aux > 0 and old_target_mandatory is not None:
                logits_diff = mandatory_logits[:, :, 1] - mandatory_logits[:, :, 0]
                pos_weight = torch.tensor([50.0], device=self.device)
                loss_mandatory_aux = F.binary_cross_entropy_with_logits(
                    logits_diff, old_target_mandatory, pos_weight=pos_weight
                )
                loss += self.lambda_mask_aux * loss_mandatory_aux

            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()

            mean_loss += loss.item()

        # Clear memory after update
        self.manager.clear_memory()

        # Log detailed HRL metrics (from last PPO epoch)
        self.log("train/manager_loss", mean_loss / self.ppo_epochs)
        self.log("train/manager_actor_loss", actor_loss)
        self.log("train/manager_value_loss", value_loss)
        self.log("train/manager_entropy", entropy)
        self.log("train/manager_advantage", advantages.mean())
        self.log("train/manager_credit_weight", credit_weight.mean())
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
        self.log("train/manager_clip_fraction", clip_fraction)

    def configure_optimizers(self):
        """
        Configure optimizer for manager parameters.

        Returns:
            Adam optimizer for manager network.
        """
        # We only train Manager with this PPO logic usually?
        # worker is usually pre-trained or trained separately?
        # In manager_train.py, it updated MANAGER.
        # If we want to train worker too, it should be in the loop or separate.
        # Here we follow manager_train.py: train manager.
        return torch.optim.Adam(self.manager.parameters(), lr=self.lr)

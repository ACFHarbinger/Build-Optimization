"""
DR-ALNS (Deep Reinforcement Learning - Adaptive Large Neighborhood Search)
PyTorch Lightning Module.

Integrates DR-ALNS with the existing RL training pipeline using Gymnasium
environments and PPO training.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam

from logic.src.envs.dr_alns import DRALNSEnv
from logic.src.models.core.dr_alns import DRALNSPPOAgent
from logic.src.tracking.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    pass

logger = get_pylogger(__name__)


class DRALNSLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for training DR-ALNS with PPO.

    This module wraps the DR-ALNS PPO agent and Gymnasium environment
    into the Lightning training framework.
    """

    def __init__(
        self,
        env: DRALNSEnv,
        agent: DRALNSPPOAgent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        n_steps_per_epoch: int = 2048,
        batch_size: int = 64,
        instance_generator: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize DR-ALNS Lightning module.

        Args:
            env: DR-ALNS Gymnasium environment.
            agent: PPO agent network.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda for advantage estimation.
            clip_epsilon: PPO clipping parameter.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_epochs: Number of PPO update epochs per batch.
            n_steps_per_epoch: Steps to collect per training epoch.
            batch_size: Minibatch size for updates.
            instance_generator: Function to generate problem instances.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["env", "agent", "instance_generator"])

        self.env = env
        self.agent = agent
        self.instance_generator = instance_generator

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.batch_size = batch_size

        # Experience buffer (reset each epoch)
        self.reset_buffer()
        self.automatic_optimization = False

    def reset_buffer(self) -> None:
        """Reset the experience buffer."""
        self.buffer_states: List[torch.Tensor] = []
        self.buffer_actions: List[Dict[str, int]] = []
        self.buffer_log_probs: List[Dict[str, torch.Tensor]] = []
        self.buffer_rewards: List[float] = []
        self.buffer_dones: List[bool] = []
        self.buffer_values: List[torch.Tensor] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        PyTorch Lightning training step.

        Note: For DR-ALNS, we don't use batches in the traditional sense.
        Instead, we collect experience by running the environment.

        Args:
            batch: Unused (required by Lightning interface).
            batch_idx: Batch index (unused).

        Returns:
            Total loss tensor.
        """
        # Collect experience
        self.collect_experience(self.n_steps_per_epoch)

        # Compute advantages and returns
        states, actions, log_probs, advantages, returns = self._process_buffer()

        # PPO update for n_epochs
        total_policy_loss: float = 0.0
        total_value_loss: float = 0.0
        total_entropy: float = 0.0
        n_updates: int = 0

        for _epoch in range(self.n_epochs):
            # Create minibatches
            n_samples = len(states)
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = states[mb_indices]
                mb_actions = {k: v[mb_indices] for k, v in actions.items()}
                mb_old_log_probs = {k: v[mb_indices] for k, v in log_probs.items()}
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Evaluate actions
                new_log_probs, new_values, entropies = self.agent.evaluate_actions(mb_states, mb_actions)

                # Compute ratios
                ratios = {}
                for key in new_log_probs.keys():
                    ratios[key] = torch.exp(new_log_probs[key] - mb_old_log_probs[key])

                # Average ratio across action heads
                avg_ratio = torch.stack(list(ratios.values())).mean(dim=0)

                # PPO policy loss
                surr1 = avg_ratio * mb_advantages
                surr2 = torch.clamp(avg_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                # Entropy bonus
                avg_entropy = torch.stack(list(entropies.values())).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * avg_entropy

                # Backward pass (manual optimization)
                self.manual_backward(loss)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += avg_entropy.item()
                n_updates += 1

        # Average losses
        mean_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0.0
        mean_value_loss = total_value_loss / n_updates if n_updates > 0 else 0.0
        mean_entropy = total_entropy / n_updates if n_updates > 0 else 0.0

        # Log metrics
        self.log("train/policy_loss", mean_policy_loss, prog_bar=True)
        self.log("train/value_loss", mean_value_loss, prog_bar=True)
        self.log("train/entropy", mean_entropy)

        # Return total loss
        return torch.tensor(
            mean_policy_loss + self.value_loss_coef * mean_value_loss - self.entropy_coef * mean_entropy
        )

    def collect_experience(self, n_steps: int):
        """
        Collect experience by running the agent in the environment.

        Args:
            n_steps: Number of steps to collect.
        """
        self.agent.eval()

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_best_profit = info.get("best_profit", 0.0)
        episode_count = 0

        for _step in range(n_steps):
            # Convert observation to tensor
            state_tensor = torch.from_numpy(obs).float().to(self.device)

            # Get action from agent
            with torch.no_grad():
                actions, log_probs, value = self.agent.get_action(state_tensor)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(
                np.array(
                    [
                        actions["destroy"],
                        actions["repair"],
                        actions["severity"],
                        actions["temp"],
                    ]
                )
            )

            done = terminated or truncated
            episode_reward += reward
            episode_best_profit = max(episode_best_profit, info.get("best_profit", 0.0))

            # Store transition
            self.buffer_states.append(state_tensor)
            self.buffer_actions.append(actions)
            self.buffer_log_probs.append(log_probs)
            self.buffer_rewards.append(reward)
            self.buffer_dones.append(done)
            self.buffer_values.append(value)

            obs = next_obs

            if done:
                # Episode finished - log and reset
                self.log("train/episode_reward", episode_reward)
                self.log("train/episode_best_profit", episode_best_profit)
                episode_count += 1

                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_best_profit = info.get("best_profit", 0.0)

        if episode_count > 0:
            self.log("train/episodes_per_epoch", float(episode_count))

    def _process_buffer(
        self,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Process the experience buffer to compute advantages and returns.

        Returns:
            Tuple of (states, actions, log_probs, advantages, returns).
        """
        # Stack experiences
        states = torch.stack(self.buffer_states).to(self.device)

        actions = {
            "destroy": torch.tensor(
                [a["destroy"] for a in self.buffer_actions],
                device=self.device,
                dtype=torch.long,
            ),
            "repair": torch.tensor(
                [a["repair"] for a in self.buffer_actions],
                device=self.device,
                dtype=torch.long,
            ),
            "severity": torch.tensor(
                [a["severity"] for a in self.buffer_actions],
                device=self.device,
                dtype=torch.long,
            ),
            "temp": torch.tensor(
                [a["temp"] for a in self.buffer_actions],
                device=self.device,
                dtype=torch.long,
            ),
        }

        log_probs = {
            "destroy": torch.stack([lp["destroy"] for lp in self.buffer_log_probs]).to(self.device),
            "repair": torch.stack([lp["repair"] for lp in self.buffer_log_probs]).to(self.device),
            "severity": torch.stack([lp["severity"] for lp in self.buffer_log_probs]).to(self.device),
            "temp": torch.stack([lp["temp"] for lp in self.buffer_log_probs]).to(self.device),
        }

        rewards = torch.tensor(self.buffer_rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.buffer_dones, device=self.device, dtype=torch.float32)
        values = torch.stack(self.buffer_values).to(self.device).squeeze(-1)

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae: torch.Tensor = torch.tensor(0.0, device=self.device)
        next_value: torch.Tensor = torch.tensor(0.0, device=self.device)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, log_probs, advantages, returns

    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        optimizer = Adam(self.agent.parameters(), lr=self.lr)
        return optimizer

"""
PPO Trainer for DR-ALNS.

Training loop for PPO agent using the DR-ALNS Gymnasium environment.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

if TYPE_CHECKING:
    from logic.src.envs.dr_alns import DRALNSEnv

from .ppo_agent import DRALNSPPOAgent


class PPOBuffer:
    """
    Experience buffer for PPO training.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.reset()

    def reset(self) -> None:
        """Clear the buffer."""
        self.states: List[torch.Tensor] = []
        self.actions: List[Dict[str, int]] = []
        self.log_probs: List[Dict[str, torch.Tensor]] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def add(
        self,
        state: torch.Tensor,
        action: Dict[str, int],
        log_prob: Dict[str, torch.Tensor],
        reward: float,
        done: bool,
        value: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self, device: torch.device) -> Dict[str, Any]:
        """
        Get all experiences as tensors.

        Returns:
            Dictionary containing batched tensors.
        """
        states = torch.stack(self.states).to(device)

        # Stack actions
        actions_dict = {
            "destroy": torch.tensor([a["destroy"] for a in self.actions], device=device, dtype=torch.long),
            "repair": torch.tensor([a["repair"] for a in self.actions], device=device, dtype=torch.long),
            "severity": torch.tensor([a["severity"] for a in self.actions], device=device, dtype=torch.long),
            "temp": torch.tensor([a["temp"] for a in self.actions], device=device, dtype=torch.long),
        }

        # Stack log probs
        log_probs_dict = {
            "destroy": torch.stack([lp["destroy"] for lp in self.log_probs]).to(device),
            "repair": torch.stack([lp["repair"] for lp in self.log_probs]).to(device),
            "severity": torch.stack([lp["severity"] for lp in self.log_probs]).to(device),
            "temp": torch.stack([lp["temp"] for lp in self.log_probs]).to(device),
        }

        rewards = torch.tensor(self.rewards, device=device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=device, dtype=torch.float32)
        values = torch.stack(self.values).to(device).squeeze(-1)

        return {
            "states": states,
            "actions": actions_dict,
            "log_probs": log_probs_dict,
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }

    def __len__(self) -> int:
        return len(self.states)


class PPOTrainer:
    """
    PPO Trainer for DR-ALNS agent.
    """

    def __init__(
        self,
        agent: DRALNSPPOAgent,
        env: DRALNSEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            agent: PPO agent to train.
            env: DR-ALNS environment.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda for advantage estimation.
            clip_epsilon: PPO clipping parameter.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_epochs: Number of PPO update epochs per batch.
            batch_size: Minibatch size for updates.
            device: Torch device.
        """
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        self.agent.to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=lr)

        self.buffer = PPOBuffer()

    def collect_experience(
        self, n_steps: int, instance_generator: Optional[Callable[..., Any]] = None
    ) -> Dict[str, float]:
        """
        Collect experience by running the agent in the environment.

        Args:
            n_steps: Number of steps to collect.
            instance_generator: Optional instance generator for the environment.

        Returns:
            Dictionary with collection statistics.
        """
        self.agent.eval()
        self.buffer.reset()

        episode_rewards = []
        episode_best_profits = []

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_best_profit = info["best_profit"]

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
            episode_best_profit = max(episode_best_profit, info["best_profit"])

            # Store transition
            self.buffer.add(state_tensor, actions, log_probs, reward, done, value)

            obs = next_obs

            if done:
                # Episode finished
                episode_rewards.append(episode_reward)
                episode_best_profits.append(episode_best_profit)

                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_best_profit = info.get("best_profit", 0.0)

        stats = {
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_best_profit": float(np.mean(episode_best_profits)) if episode_best_profits else 0.0,
            "n_episodes": float(len(episode_rewards)),
        }

        return stats

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor (n_steps,).
            values: Value tensor (n_steps,).
            dones: Done flags (n_steps,).

        Returns:
            Tuple of (advantages, returns).
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae: Union[float, torch.Tensor] = 0.0
        next_value: Union[float, torch.Tensor] = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Update the agent using collected experience.

        Returns:
            Dictionary with training statistics.
        """
        self.agent.train()

        # Get all experiences
        batch = self.buffer.get(self.device)

        # Compute advantages
        advantages, returns = self.compute_advantages(batch["rewards"], batch["values"], batch["dones"])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        n_samples = len(batch["states"])

        for _epoch in range(self.n_epochs):
            # Create minibatches
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = batch["states"][mb_indices]
                actions_batch = cast(Dict[str, torch.Tensor], batch["actions"])
                log_probs_batch = cast(Dict[str, torch.Tensor], batch["log_probs"])
                mb_actions = {k: v[mb_indices] for k, v in actions_batch.items()}
                mb_old_log_probs = {k: v[mb_indices] for k, v in log_probs_batch.items()}
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Evaluate actions
                new_log_probs, new_values, entropies = self.agent.evaluate_actions(mb_states, mb_actions)

                # Compute ratios for each action head
                ratios = {}
                for key in new_log_probs.keys():
                    ratios[key] = torch.exp(new_log_probs[key] - mb_old_log_probs[key])

                # Average ratio across action heads
                avg_ratio = torch.stack(list(ratios.values())).mean(dim=0)

                # PPO policy loss (average across action heads)
                surr1 = avg_ratio * mb_advantages
                surr2 = torch.clamp(avg_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                # Entropy bonus (average across action heads)
                avg_entropy = torch.stack(list(entropies.values())).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * avg_entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += avg_entropy.item()

        n_updates = self.n_epochs * (n_samples // self.batch_size)
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

        return stats

    def train(
        self,
        total_timesteps: int,
        n_steps_per_update: int = 2048,
        log_interval: int = 10,
        instance_generator: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train.
            n_steps_per_update: Number of steps to collect before update.
            log_interval: How often to log (in updates).
            instance_generator: Optional instance generator.

        Returns:
            Dictionary with training history.
        """
        history: Dict[str, List[float]] = {
            "mean_episode_reward": [],
            "mean_best_profit": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        n_updates = total_timesteps // n_steps_per_update

        for update in range(n_updates):
            # Collect experience
            collect_stats = self.collect_experience(n_steps_per_update, instance_generator)

            # Update agent
            update_stats = self.update()

            # Log
            if (update + 1) % log_interval == 0:
                print(
                    f"Update {update + 1}/{n_updates} | "
                    f"Reward: {collect_stats['mean_episode_reward']:.2f} | "
                    f"Best Profit: {collect_stats['mean_best_profit']:.2f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                    f"Value Loss: {update_stats['value_loss']:.4f}"
                )

            # Store history
            history["mean_episode_reward"].append(collect_stats["mean_episode_reward"])
            history["mean_best_profit"].append(collect_stats["mean_best_profit"])
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])

        return history

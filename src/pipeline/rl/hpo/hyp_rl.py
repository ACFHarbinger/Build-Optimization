"""
Hyp-RL: Hyperparameter Optimization by Reinforcement Learning.

This module implements the Hyp-RL method from the paper:
"Hyp-RL: Hyperparameter Optimization by Reinforcement Learning" (Jomaa et al., 2019)

The method formulates HPO as a sequential decision problem where an RL agent
learns to select which hyperparameter configuration to evaluate next, based on
the history of previous evaluations and their rewards.

Key Components:
    - State: History of hyperparameter configurations and their performance
    - Action: Next hyperparameter configuration to evaluate
    - Reward: Improvement in validation performance
    - Policy Network: LSTM-based network that processes history and generates configs
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from logic.src.configs import Config

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]

from .base import BaseHPO, ParamSpec, apply_params
from .hyp_rl_pol import HypRLPolicy


class HypRLHPO(BaseHPO):
    """
    Hyp-RL: Hyperparameter Optimization by Reinforcement Learning.

    Uses an LSTM-based policy network to learn which hyperparameter
    configurations to evaluate next, based on the history of evaluations.

    The policy is trained with REINFORCE using the improvement in validation
    performance as the reward signal.
    """

    def __init__(
        self,
        cfg: Config,
        objective_fn: Callable,
        search_space: Optional[Dict[str, ParamSpec]] = None,
    ):
        """Initialize Hyp-RL HPO.

        Args:
            cfg: Root application configuration.
            objective_fn: Callable that trains a model and returns metric to maximize.
            search_space: Optional pre-normalized search space.
        """
        super().__init__(cfg, objective_fn, search_space)

        # Hyp-RL specific configuration
        self.state_dim = 64
        self.hidden_dim = 128
        self.n_layers = 2
        self.policy_lr = 1e-3
        self.entropy_weight = 0.01
        self.gamma = 0.99  # Discount factor

        # Build policy network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = HypRLPolicy(
            search_space=self.search_space,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        # Episode memory
        self.episode_configs: List[Dict[str, Any]] = []
        self.episode_log_probs: List[Dict[str, torch.Tensor]] = []
        self.episode_rewards: List[float] = []

        # Best configuration tracking
        self.best_value = float("-inf")
        self.best_config: Dict[str, Any] = {}

    def run(self) -> float:
        """Run the Hyp-RL optimization.

        Returns:
            The best metric value found across all trials.
        """
        n_trials = self.cfg.hpo.n_trials
        history: List[Tuple[Dict[str, Any], float]] = []
        hidden = None

        for trial_idx in range(n_trials):
            # Generate configuration using policy
            sampled_config, log_probs, hidden = self.policy(history, hidden)

            # Apply configuration to cfg
            trial_cfg = apply_params(self.cfg, sampled_config)

            # Evaluate configuration
            metric_value = self.objective_fn(trial_idx, trial_cfg)

            # Store in history
            history.append((sampled_config, metric_value))

            # Store in episode memory
            self.episode_configs.append(sampled_config)
            self.episode_log_probs.append(log_probs)
            self.episode_rewards.append(metric_value)

            # Track best
            if metric_value > self.best_value:
                self.best_value = metric_value
                self.best_config = sampled_config.copy()

            # Update policy periodically
            if (trial_idx + 1) % 10 == 0 or trial_idx == n_trials - 1:
                self._update_policy()
                self._reset_episode()

        # Log to tracking system
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params({f"hpo/best/{k}": v for k, v in self.best_config.items()})
            run.log_metric("hpo/best_value", self.best_value)
            run.log_metric("hpo/n_trials", n_trials)
            run.set_tag("hpo_backend", "hyp_rl")
            run.set_tag("hpo_method", "rl")

        return self.best_value

    def _update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return

        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_losses = []

        for log_probs_dict, G in zip(self.episode_log_probs, returns, strict=False):
            # Sum log probs across all parameters
            total_log_prob = sum(log_probs_dict.values())

            # REINFORCE: -log_prob * return
            policy_loss = -total_log_prob * G
            policy_losses.append(policy_loss)

        # Average loss
        total_loss = torch.stack(policy_losses).mean()

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _reset_episode(self):
        """Reset episode memory."""
        self.episode_configs = []
        self.episode_log_probs = []
        self.episode_rewards = []

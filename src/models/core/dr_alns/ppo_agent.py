"""
PPO Agent for DR-ALNS.

Multi-headed actor-critic network for controlling ALNS parameters online.
Uses PPO (Proximal Policy Optimization) as described in the DR-ALNS paper.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DRALNSPPOAgent(nn.Module):
    """
    PPO Agent for DR-ALNS with multiple discrete action heads.

    The agent learns to control ALNS by selecting:
    1. Destroy operator
    2. Repair operator
    3. Destroy severity (percentage)
    4. Acceptance criterion temperature

    Architecture: MLP with 2 hidden layers of size 64,
    separate output heads for each discrete action.
    """

    def __init__(
        self,
        state_dim: int = 7,
        hidden_dim: int = 64,
        n_destroy_ops: int = 3,
        n_repair_ops: int = 3,
        n_severity_levels: int = 10,
        n_temp_levels: int = 50,
    ):
        """
        Initialize the PPO agent.

        Args:
            state_dim: Dimension of state vector (default 7 per paper).
            hidden_dim: Hidden layer size (default 64 per paper).
            n_destroy_ops: Number of destroy operators.
            n_repair_ops: Number of repair operators.
            n_severity_levels: Number of destroy severity levels (1-10 = 10%-100%).
            n_temp_levels: Number of temperature levels (1-50 = 0.1-5.0).
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_destroy_ops = n_destroy_ops
        self.n_repair_ops = n_repair_ops
        self.n_severity_levels = n_severity_levels
        self.n_temp_levels = n_temp_levels

        # Shared feature extractor (2 hidden layers)
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor heads (one for each discrete action)
        self.destroy_head = nn.Linear(hidden_dim, n_destroy_ops)
        self.repair_head = nn.Linear(hidden_dim, n_repair_ops)
        self.severity_head = nn.Linear(hidden_dim, n_severity_levels)
        self.temp_head = nn.Linear(hidden_dim, n_temp_levels)

        # Critic head (value function)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: State tensor of shape (batch, state_dim).

        Returns:
            Tuple of (destroy_logits, repair_logits, severity_logits, temp_logits, value).
        """
        features = self.shared_net(state)

        destroy_logits = self.destroy_head(features)
        repair_logits = self.repair_head(features)
        severity_logits = self.severity_head(features)
        temp_logits = self.temp_head(features)
        value = self.value_head(features)

        return destroy_logits, repair_logits, severity_logits, temp_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Sample actions from the policy.

        Args:
            state: State tensor of shape (batch, state_dim) or (state_dim,).
            deterministic: If True, select argmax instead of sampling.

        Returns:
            Tuple of (actions_dict, log_probs_dict, value).
            - actions_dict: {"destroy": int, "repair": int, "severity": int, "temp": int}
            - log_probs_dict: {"destroy": tensor, "repair": tensor, ...}
            - value: State value estimate
        """
        # Handle single state (unbatched)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        destroy_logits, repair_logits, severity_logits, temp_logits, value = self(state)

        # Create categorical distributions
        destroy_dist = Categorical(logits=destroy_logits)
        repair_dist = Categorical(logits=repair_logits)
        severity_dist = Categorical(logits=severity_logits)
        temp_dist = Categorical(logits=temp_logits)

        # Sample or select argmax
        if deterministic:
            destroy_action = destroy_logits.argmax(dim=-1)
            repair_action = repair_logits.argmax(dim=-1)
            severity_action = severity_logits.argmax(dim=-1)
            temp_action = temp_logits.argmax(dim=-1)
        else:
            destroy_action = destroy_dist.sample()
            repair_action = repair_dist.sample()
            severity_action = severity_dist.sample()
            temp_action = temp_dist.sample()

        # Compute log probabilities
        destroy_log_prob = destroy_dist.log_prob(destroy_action)
        repair_log_prob = repair_dist.log_prob(repair_action)
        severity_log_prob = severity_dist.log_prob(severity_action)
        temp_log_prob = temp_dist.log_prob(temp_action)

        if squeeze_output:
            actions = {
                "destroy": destroy_action.item(),
                "repair": repair_action.item(),
                "severity": severity_action.item(),
                "temp": temp_action.item(),
            }
            log_probs = {
                "destroy": destroy_log_prob.squeeze(0),
                "repair": repair_log_prob.squeeze(0),
                "severity": severity_log_prob.squeeze(0),
                "temp": temp_log_prob.squeeze(0),
            }
            value = value.squeeze(0)
        else:
            actions = {
                "destroy": destroy_action,
                "repair": repair_action,
                "severity": severity_action,
                "temp": temp_action,
            }
            log_probs = {
                "destroy": destroy_log_prob,
                "repair": repair_log_prob,
                "severity": severity_log_prob,
                "temp": temp_log_prob,
            }

        return actions, log_probs, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions taken in given states (for PPO update).

        Args:
            states: State tensor of shape (batch, state_dim).
            actions: Dictionary of action tensors.

        Returns:
            Tuple of (log_probs_dict, values, entropy_dict).
        """
        destroy_logits, repair_logits, severity_logits, temp_logits, values = self(states)

        # Create distributions
        destroy_dist = Categorical(logits=destroy_logits)
        repair_dist = Categorical(logits=repair_logits)
        severity_dist = Categorical(logits=severity_logits)
        temp_dist = Categorical(logits=temp_logits)

        # Compute log probabilities
        log_probs = {
            "destroy": destroy_dist.log_prob(actions["destroy"]),
            "repair": repair_dist.log_prob(actions["repair"]),
            "severity": severity_dist.log_prob(actions["severity"]),
            "temp": temp_dist.log_prob(actions["temp"]),
        }

        # Compute entropies (for entropy regularization)
        entropies = {
            "destroy": destroy_dist.entropy(),
            "repair": repair_dist.entropy(),
            "severity": severity_dist.entropy(),
            "temp": temp_dist.entropy(),
        }

        return log_probs, values, entropies


class DRALNSState:
    """
    State representation for DR-ALNS.

    Based on the 7 problem-agnostic features from the paper:
    1. Best improved (0/1)
    2. Current accepted (0/1)
    3. Current improved (0/1)
    4. Is current best (0/1)
    5. Cost difference from best (percentage, -1 if objective <= 0)
    6. Stagnation count (number of iterations without improvement)
    7. Search budget used (percentage)
    """

    def __init__(self):
        self.best_improved = 0
        self.current_accepted = 0
        self.current_improved = 0
        self.is_current_best = 0
        self.cost_diff_best = 0.0
        self.stagnation_count = 0
        self.search_budget = 0.0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert state to tensor."""
        state_vec = torch.tensor(
            [
                float(self.best_improved),
                float(self.current_accepted),
                float(self.current_improved),
                float(self.is_current_best),
                self.cost_diff_best,
                float(self.stagnation_count),
                self.search_budget,
            ],
            dtype=torch.float32,
        )
        if device is not None:
            state_vec = state_vec.to(device)
        return state_vec

    def update(
        self,
        best_profit: float,
        current_profit: float,
        previous_profit: float,
        new_accepted: bool,
        new_best: bool,
        iteration: int,
        max_iterations: int,
        iterations_since_best: int,
    ):
        """
        Update state based on ALNS search dynamics.

        Args:
            best_profit: Best profit found so far.
            current_profit: Current solution profit.
            previous_profit: Previous iteration profit.
            new_accepted: Whether new solution was accepted.
            iteration: Current iteration number.
            max_iterations: Maximum number of iterations.
            iterations_since_best: Iterations since last best solution.
        """
        # Binary features
        self.best_improved = 1 if new_best else 0
        self.current_accepted = 1 if new_accepted else 0
        self.current_improved = 1 if (new_accepted and current_profit > previous_profit) else 0
        self.is_current_best = 1 if abs(current_profit - best_profit) < 1e-6 else 0

        # Cost difference from best (as percentage)
        if current_profit > 0:
            self.cost_diff_best = (best_profit - current_profit) / abs(current_profit)
        else:
            self.cost_diff_best = -1.0

        # Stagnation count
        self.stagnation_count = iterations_since_best

        # Search budget used (percentage)
        self.search_budget = iteration / max_iterations if max_iterations > 0 else 0.0

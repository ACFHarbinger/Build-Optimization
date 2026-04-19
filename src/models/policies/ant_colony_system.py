"""ant_colony_system.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import ant_colony_system
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedACOPolicy(AutoregressivePolicy, PolicyVizMixin):
    """
    Vectorized Ant Colony Optimization (ACO) Policy.

    Implements a tensor-based ACO that processes multiple batch instances
    and multiple ants per instance in parallel on the GPU.
    """

    def __init__(
        self,
        env_name: str,
        n_ants: int = 20,
        n_iterations: int = 50,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,  # Heuristic importance
        decay: float = 0.1,  # Pheromone evaporation rate (rho)
        elitism: int = 1,  # Number of best ants to use for pheromone update
        q0: float = 0.9,  # Probability of exploiting best edge (ACS)
        min_pheromone: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize Class.

        Args:
            env_name (str): Description of env_name.
            n_ants (int): Description of n_ants.
            n_iterations (int): Description of n_iterations.
            alpha (float): Description of alpha.
            beta (float): Description of beta.
            decay (float): Description of decay.
            elitism (int): Description of elitism.
            q0 (float): Description of q0.
            min_pheromone (float): Description of min_pheromone.
            seed (int): Description of seed.
            device (str): Description of device.
            kwargs (Any): Description of kwargs.
        """
        super().__init__(env_name=env_name, seed=seed, device=device, **kwargs)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.seed = seed
        self.beta = beta
        self.decay = decay
        self.elitism = elitism
        self.q0 = q0
        self.min_pheromone = min_pheromone

    def __getstate__(self) -> Dict[str, Any]:
        """Return state for pickling."""
        state = self.__dict__.copy()
        # Handle torch.Generator which is not picklable
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after pickling."""
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            # Create a new generator on the same device
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def _get_heuristic(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Compute heuristic information (eta = 1 / distance)."""
        # Avoid division by zero
        dist_safe = dist_matrix + 1e-6
        return 1.0 / dist_safe

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "sampling",  # Unused
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run vectorized ACO.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # 1. Setup Data
        locs = td["locs"]
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        num_nodes = dist_matrix.shape[1]

        # Heuristic (eta)
        eta = self._get_heuristic(dist_matrix)

        # Initialize Pheromone (tau)
        tau = torch.ones_like(dist_matrix) * self.min_pheromone

        # Best solution tracking
        best_tours = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=device)
        best_costs = torch.full((batch_size,), float("inf"), device=device)

        # 2. Main ACO Loop
        for _aco_iter in range(self.n_iterations):
            # Construct solutions for all ants: [B, n_ants, N]
            iter_ant_tours, _ = self._construct_solutions(dist_matrix, tau, eta, env)

            # Calculate costs: [B, n_ants]
            costs = self._evaluate_batch(iter_ant_tours, dist_matrix)

            # Update best found so far
            min_costs, min_idx = costs.min(dim=1)
            improved = min_costs < best_costs
            best_costs[improved] = min_costs[improved]

            # Indexing to get best tours: [B, N]
            best_ants_indices = min_idx.view(batch_size, 1, 1).expand(-1, 1, num_nodes)
            best_tours_iter = iter_ant_tours.gather(1, best_ants_indices).squeeze(1)
            best_tours[improved] = best_tours_iter[improved]

            # Update Pheromones
            tau = self._update_pheromones(tau, iter_ant_tours, costs)

            self._viz_record(
                iteration=_aco_iter,
                global_best_cost=float(best_costs.min().item()),
                iter_best_cost=float(min_costs.min().item()),
                tau_mean=float(tau.mean().item()),
                tau_max=float(tau.max().item()),
            )

        # Return result
        return {
            "actions": best_tours,
            "reward": -best_costs,
            "cost": best_costs,
        }

    def _construct_solutions(
        self, dist_matrix: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor, env: Optional[RL4COEnvBase]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct routes for all ants in parallel.
        """
        batch_size, num_nodes, _ = dist_matrix.shape
        n_ants = self.n_ants
        device = dist_matrix.device

        # Expand for ants: [B, K, N, N]
        tau_k = tau.unsqueeze(1).expand(-1, n_ants, -1, -1)
        eta_k = eta.unsqueeze(1).expand(-1, n_ants, -1, -1)

        # Start at node 0 (depot)
        current_node = torch.zeros((batch_size, n_ants), dtype=torch.long, device=device)
        visited = torch.zeros((batch_size, n_ants, num_nodes), dtype=torch.bool, device=device)
        visited.scatter_(2, current_node.unsqueeze(2), 1)

        tours_accumulator: list[torch.Tensor] = [current_node]
        log_probs_sum = torch.zeros((batch_size, n_ants), device=device)

        # Step through nodes
        for _ in range(num_nodes - 1):
            # Get tau and eta for current edges
            gather_idx = current_node.view(batch_size, n_ants, 1, 1).expand(-1, -1, 1, num_nodes)
            tau_step = tau_k.gather(2, gather_idx).squeeze(2)
            eta_step = eta_k.gather(2, gather_idx).squeeze(2)

            scores = (tau_step**self.alpha) * (eta_step**self.beta)
            scores.masked_fill_(visited, 0)

            # Selection
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-10)

            rand_val = torch.rand((batch_size, n_ants), generator=self.generator, device=device)
            greedy_mask = rand_val < self.q0
            greedy_action = probs.argmax(dim=-1)
            sample_probs = probs + 1e-10
            sample_action = torch.multinomial(sample_probs.view(-1, num_nodes), 1, generator=self.generator).view(
                batch_size, n_ants
            )

            next_node = torch.where(greedy_mask, greedy_action, sample_action)

            # Update state
            current_node = next_node
            visited.scatter_(2, current_node.unsqueeze(2), 1)
            tours_accumulator.append(current_node)

            # Track log probs
            selected_probs = probs.gather(2, next_node.unsqueeze(2)).squeeze(2)
            log_probs_sum += torch.log(selected_probs + 1e-10)

        final_ant_tours = torch.stack(tours_accumulator, dim=2)
        return final_ant_tours, log_probs_sum

    def _evaluate_batch(self, ant_tours: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Calculate lengths of all ant tours."""
        batch_size, n_ants, num_nodes = ant_tours.shape

        from_node = ant_tours
        to_node = torch.roll(ant_tours, -1, dims=2)

        batch_idx = (
            torch.arange(batch_size, device=ant_tours.device)
            .view(batch_size, 1, 1)
            .expand(batch_size, n_ants, num_nodes)
        )
        dists = dist_matrix[batch_idx, from_node, to_node]

        return dists.sum(dim=2)

    def _update_pheromones(self, tau: torch.Tensor, ant_tours: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """Update pheromone matrix."""
        tau = (1 - self.decay) * tau
        batch_size, n_ants, num_nodes = ant_tours.shape

        min_costs, min_idx = costs.min(dim=1)
        idx_expanded = min_idx.view(batch_size, 1, 1).expand(-1, 1, num_nodes)
        best_tours = ant_tours.gather(1, idx_expanded).squeeze(1)

        delta = 1.0 / (min_costs.view(batch_size, 1, 1) + 1e-10)

        from_node = best_tours
        to_node = torch.roll(best_tours, -1, dims=1)
        batch_idx = torch.arange(batch_size, device=tau.device).view(batch_size, 1)

        b_idx_flat = batch_idx.expand(-1, num_nodes).reshape(-1)
        from_flat = from_node.reshape(-1)
        to_flat = to_node.reshape(-1)
        delta_flat = delta.expand(-1, -1, num_nodes).reshape(-1)

        tau.index_put_((b_idx_flat, from_flat, to_flat), delta_flat, accumulate=True)
        tau.index_put_((b_idx_flat, to_flat, from_flat), delta_flat, accumulate=True)

        return torch.clamp(tau, min=self.min_pheromone, max=10.0)

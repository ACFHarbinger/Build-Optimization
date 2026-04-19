"""
ACO Decoder: Ant colony solution construction from heatmaps.

Uses pheromone-guided ant colony optimization to construct solutions.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.decoder import NonAutoregressiveDecoder


class ACODecoder(NonAutoregressiveDecoder):
    """
    Ant Colony Optimization decoder.

    Constructs solutions using heatmap-guided pheromones.
    """

    def __init__(
        self,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize ACODecoder.

        Args:
            n_ants: Number of ants per construction.
            n_iterations: Number of ACO iterations.
            alpha: Pheromone importance weight.
            beta: Heuristic (distance) importance weight.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to apply 2-opt local search.
        """
        super().__init__(**kwargs)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.use_local_search = use_local_search
        self.seed = seed
        self.init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=self.init_device).manual_seed(self.seed)

    def __getstate__(self):
        """Prepare state for pickling (handle non-picklable Generator)."""
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce logits for the current step (for Trajectory Balance training).
        """
        current_node = td.get("current_node", torch.zeros(td.batch_size, dtype=torch.long, device=heatmap.device))
        # heatmap: [batch, n, n]
        logits = heatmap.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, heatmap.size(-1))).squeeze(1)
        return logits, td["mask"]

    def construct(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        num_starts: int = 1,
        return_all: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct solutions using ACO with heatmap pheromones.

        Args:
            td: TensorDict with problem instance.
            heatmap: Edge heatmap [batch, n, n] (log probabilities).
            env: Environment for reward calculation.
            num_starts: Number of independent ACO runs.

        Returns:
            Dictionary with actions, log_likelihood, reward.
        """
        batch_size, num_nodes, _ = heatmap.shape

        # Convert log heatmap to pheromone
        pheromone = heatmap.exp()  # [batch, n, n]

        # Compute distance matrix for heuristic
        locs = td["locs"]  # [batch, n, 2]
        dist_matrix = torch.cdist(locs, locs, p=2)  # [batch, n, n]
        # Heuristic: inverse distance (avoid div by zero)
        heuristic = 1.0 / (dist_matrix + 1e-10)

        # Combined probability matrix
        prob_matrix = (pheromone**self.alpha) * (heuristic**self.beta)

        # Run ants
        if return_all:
            all_tours, all_costs, all_log_probs, best_tours, best_costs, _ = self._run_ants(  # type: ignore[misc]
                prob_matrix, dist_matrix, td, env, return_all=True
            )

            # Optional local search
            out = {
                "actions": all_tours,  # (batch, n_ants, n_nodes)
                "reward": -all_costs,  # (batch, n_ants)
                "log_likelihood": all_log_probs,  # (batch, n_ants)
            }

            if self.use_local_search:
                # Apply LS to all ants
                # Reshape for _two_opt: (B, A, N) -> (B*A, N)
                b, a, n = all_tours.shape
                flat_tours = all_tours.reshape(b * a, n)

                # Expand dist_matrix: (B, N, N) -> (B*A, N, N)
                flat_dist = dist_matrix.unsqueeze(1).repeat(1, a, 1, 1).reshape(b * a, n, n)

                ls_tours, ls_costs = self._two_opt(flat_tours, flat_dist)

                out["ls_actions"] = ls_tours.reshape(b, a, n)
                out["ls_reward"] = -ls_costs.reshape(b, a)
                out["ls_log_likelihood"] = all_log_probs  # Reuse log_likelihood as approx

            return out

        best_tours, best_costs, log_probs = self._run_ants(prob_matrix, dist_matrix, td, env, return_all=False)  # type: ignore[misc]

        # Optional local search
        if self.use_local_search:
            best_tours, best_costs = self._two_opt(best_tours, dist_matrix)

        return {
            "actions": best_tours,
            "reward": -best_costs,  # Reward is negative cost
            "log_likelihood": log_probs,
        }

    def _run_ants(
        self,
        prob_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
        td: TensorDict,
        env: RL4COEnvBase,
        return_all: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Run ant colony to construct solutions.

        Args:
            prob_matrix: Combined pheromone * heuristic [batch, n, n].
            dist_matrix: Distance matrix [batch, n, n].
            td: TensorDict with problem instance.
            env: Environment.

        Returns:
            Tuple of (best_tours, best_costs, log_probs).
        """
        batch_size, num_nodes, _ = prob_matrix.shape
        device = prob_matrix.device

        best_tours = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        best_costs = torch.full((batch_size,), float("inf"), device=device)
        total_log_probs = torch.zeros(batch_size, device=device)

        all_tours_list = []
        all_costs_list = []
        all_log_probs_list = []
        for _ in range(self.n_ants):
            # Construct tour for each ant
            tours, costs, log_probs = self._construct_tour(prob_matrix, dist_matrix)

            if return_all:
                all_tours_list.append(tours)
                all_costs_list.append(costs)
                all_log_probs_list.append(log_probs)

            # Update best
            improved = costs < best_costs
            best_costs = torch.where(improved, costs, best_costs)
            best_tours = torch.where(improved.unsqueeze(-1), tours, best_tours)
            total_log_probs = total_log_probs + log_probs

        if return_all:
            all_tours = torch.stack(all_tours_list, dim=1)  # (batch, n_ants, n_nodes)
            all_costs = torch.stack(all_costs_list, dim=1)  # (batch, n_ants)
            all_log_probs = torch.stack(all_log_probs_list, dim=1)  # (batch, n_ants)
            return all_tours, all_costs, all_log_probs, best_tours, best_costs, total_log_probs / self.n_ants

        return best_tours, best_costs, total_log_probs / self.n_ants

    def _construct_tour(
        self,
        prob_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct a single tour using probabilistic selection.

        Args:
            prob_matrix: Selection probabilities [batch, n, n].
            dist_matrix: Distance matrix [batch, n, n].

        Returns:
            Tuple of (tour, cost, log_prob).
        """
        batch_size, num_nodes, _ = prob_matrix.shape
        device = prob_matrix.device

        # Start from depot (node 0)
        tour = [torch.zeros(batch_size, dtype=torch.long, device=device)]
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        visited[:, 0] = True

        total_log_prob = torch.zeros(batch_size, device=device)
        current = tour[0]

        for _ in range(num_nodes - 1):
            # Get probabilities for next node
            probs = prob_matrix[torch.arange(batch_size, device=device), current]
            probs = probs.masked_fill(visited, 0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Sample next node
            next_node = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

            # Record
            tour.append(next_node)
            visited[torch.arange(batch_size, device=device), next_node] = True
            total_log_prob = total_log_prob + torch.log(
                probs[torch.arange(batch_size, device=device), next_node] + 1e-10
            )
            current = next_node

        # Stack tour
        tour_tensor = torch.stack(tour, dim=1)  # [batch, n]

        # Compute tour cost
        cost = self._compute_tour_cost(tour_tensor, dist_matrix)

        return tour_tensor, cost, total_log_prob

    def _compute_tour_cost(
        self,
        tour: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total tour distance."""
        batch_size, num_nodes = tour.shape
        device = tour.device

        # Gather distances for each edge in tour
        from_nodes = tour
        to_nodes = torch.roll(tour, -1, dims=1)

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        costs = dist_matrix[batch_idx, from_nodes, to_nodes].sum(dim=-1)
        return costs

    def _two_opt(
        self,
        tours: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2-opt local search to improve tours.

        Args:
            tours: Tour sequences [batch, n].
            dist_matrix: Distance matrix [batch, n, n].

        Returns:
            Improved (tours, costs).
        """
        # Simple 2-opt: try all swaps, keep improvements
        batch_size, num_nodes = tours.shape

        improved = True
        best_tours = tours.clone()
        best_costs = self._compute_tour_cost(best_tours, dist_matrix)

        max_iters = 10  # Limit iterations for efficiency
        for _ in range(max_iters):
            improved = False
            for i in range(1, num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    # Reverse segment [i, j]
                    new_tours = best_tours.clone()
                    new_tours[:, i : j + 1] = torch.flip(best_tours[:, i : j + 1], dims=[1])
                    new_costs = self._compute_tour_cost(new_tours, dist_matrix)

                    # Keep if improved
                    improved_mask = new_costs < best_costs
                    if improved_mask.any():
                        improved = True
                        best_costs = torch.where(improved_mask, new_costs, best_costs)
                        best_tours = torch.where(improved_mask.unsqueeze(-1), new_tours, best_tours)

            if not improved:
                break

        return best_tours, best_costs

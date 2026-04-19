"""
Vectorized Adaptive Large Neighborhood Search (ALNS) Policy.

This module implements a vectorized version of the ALNS algorithm for
Vehicle Routing Problems (VRP), optimized for GPU execution using PyTorch.
"""

import time
from typing import List, Optional, Tuple

import torch

from logic.src.models.policies.operators import (
    vectorized_cluster_removal,
    vectorized_greedy_insertion,
    vectorized_random_removal,
    vectorized_regret_k_insertion,
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
    vectorized_worst_removal,
)
from logic.src.models.policies.shared.linear import vectorized_linear_split
from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedALNS(PolicyVizMixin):
    """
    Vectorized Adaptive Large Neighborhood Search Solver.
    """

    def __init__(
        self,
        dist_matrix,
        wastes,
        vehicle_capacity,
        time_limit=1.0,
        device: str = "cuda",
        seed: int = 42,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Initialize the Vectorized ALNS solver.

        Args:
            dist_matrix: Distance matrix [B, N, N].
            wastes: Bin wastes [B, N].
            vehicle_capacity: Vehicle capacity constraint.
            time_limit: Time limit for solving in seconds.
            device: Computation device ('cpu' or 'cuda').
            generator: PyTorch generator for random number generation.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = torch.device(device)
        self.seed = seed
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        self.destroy_ops = [vectorized_random_removal, vectorized_worst_removal, vectorized_cluster_removal]
        self.repair_ops = [vectorized_greedy_insertion, vectorized_regret_k_insertion]

        self.d_weights = torch.ones(len(self.destroy_ops), device=device)
        self.r_weights = torch.ones(len(self.repair_ops), device=device)

    def __getstate__(self):
        """Prepare state for pickling."""
        state = self.__dict__.copy()
        if "generator" in state and state["generator"] is not None:
            state["generator_state"] = state["generator"].get_state()
            state["generator_device"] = str(state["generator"].device)
            del state["generator"]
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        gen_state = state.pop("generator_state", None)
        gen_device = state.pop("generator_device", None)
        self.__dict__.update(state)
        if gen_state is not None:
            self.generator = torch.Generator(device=gen_device)
            self.generator.set_state(gen_state)
        else:
            self.generator = torch.Generator(device="cpu")

    def solve(
        self,
        initial_solutions: torch.Tensor,
        n_iterations: int = 2000,
        time_limit: Optional[float] = None,
        max_vehicles: int = 0,
        start_temp: float = 0.5,
        cooling_rate: float = 0.9995,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        High-performance ALNS Solve.
        Optimizes the giant tour as a TSP and splits only at the end.
        """
        B, N = initial_solutions.size()
        device = initial_solutions.device
        start_time = time.time()

        current_solutions = initial_solutions.clone()

        # Initial TSP costs (dist_matrix[prev, curr])
        def get_tsp_costs(tours):
            """Compute total TSP distance for each tour in the batch."""
            padded = torch.zeros(B, N + 2, dtype=torch.long, device=device)
            padded[:, 1:-1] = tours
            shifted = torch.roll(padded, shifts=-1, dims=1)
            dist = self.dist_matrix[torch.arange(B, device=device).unsqueeze(1), padded, shifted]
            return dist[:, :-1].sum(dim=1)

        current_costs = get_tsp_costs(current_solutions)
        best_solutions = current_solutions.clone()
        best_costs = current_costs.clone()

        T = torch.full((B,), float(start_temp), device=device)

        for i in range(n_iterations):
            if time_limit and (time.time() - start_time > time_limit):
                break

            # 1. Select Operators
            d_idx = int(torch.multinomial(self.d_weights, 1, generator=self.generator).item())
            r_idx = int(torch.multinomial(self.r_weights, 1, generator=self.generator).item())
            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]

            n_remove = torch.randint(
                max(1, int(N * 0.1)), max(2, int(N * 0.4) + 1), (1,), device=device, generator=self.generator
            ).item()

            # 2. Destroy & Repair (Fast TSP operators)
            if destroy_op == vectorized_random_removal:
                partial, removed = destroy_op(current_solutions, int(n_remove), generator=self.generator)  # type: ignore[operator]
            else:
                partial, removed = destroy_op(current_solutions, self.dist_matrix, n_remove, generator=self.generator)  # type: ignore[operator]

            candidate_solutions = repair_op(partial, removed, self.dist_matrix)  # type: ignore[operator]

            # 3. Fast Education (TSP Local Search)
            if i % 20 == 0:
                candidate_solutions = vectorized_two_opt(
                    candidate_solutions, self.dist_matrix, max_iterations=20, generator=self.generator
                )
                candidate_solutions = vectorized_relocate(
                    candidate_solutions, self.dist_matrix, max_iterations=10, generator=self.generator
                )
                candidate_solutions = vectorized_swap(
                    candidate_solutions, self.dist_matrix, max_iterations=10, generator=self.generator
                )
                if N > 40 and i % 100 == 0:
                    candidate_solutions = vectorized_three_opt(
                        candidate_solutions, self.dist_matrix, max_iterations=5, generator=self.generator
                    )

            # 3b. Advanced Inter-Route Education (Periodic)
            if i > 0 and i % 100 == 0:
                # 1. Split into CVRP routes (returns list of lists)
                routes_list, _ = vectorized_linear_split(
                    candidate_solutions,
                    self.dist_matrix,
                    self.wastes,
                    self.vehicle_capacity,
                    max_vehicles=max_vehicles,
                )

                # 2. Convert to padded tensor for vectorized operators
                max_r_len = max(len(r) for r in routes_list)
                routes_tensor = torch.zeros((B, max_r_len), device=device, dtype=torch.long)
                for b_idx in range(B):
                    r_nodes = torch.tensor(routes_list[b_idx], device=device, dtype=torch.long)
                    routes_tensor[b_idx, : len(r_nodes)] = r_nodes

                # 3. Apply inter-route operators
                routes_tensor = vectorized_two_opt_star(
                    routes_tensor, self.dist_matrix, max_iterations=10, generator=self.generator
                )
                routes_tensor = vectorized_swap_star(
                    routes_tensor, self.dist_matrix, max_iterations=10, generator=self.generator
                )

                # 4. Flatten back to giant tour
                flattened = []
                for b_idx in range(B):
                    # extract non-zero nodes (customers)
                    nodes = routes_tensor[b_idx]
                    mask_nz = nodes != 0
                    flat_nodes = nodes[mask_nz]
                    # Ensure it has exactly N customers (it should if split was complete)
                    if flat_nodes.size(0) < N:
                        # Re-attach missing nodes if any or pad
                        existing = set(flat_nodes.tolist())
                        all_c = set(candidate_solutions[b_idx].tolist())
                        missing = list(all_c - existing)
                        if missing:
                            flat_nodes = torch.cat([flat_nodes, torch.tensor(missing, device=device, dtype=torch.long)])
                        if flat_nodes.size(0) < N:
                            flat_nodes = torch.cat(
                                [flat_nodes, torch.zeros(N - flat_nodes.size(0), device=device, dtype=torch.long)]
                            )
                    elif flat_nodes.size(0) > N:
                        flat_nodes = flat_nodes[:N]
                    flattened.append(flat_nodes)
                candidate_solutions = torch.stack(flattened)

            # 4. Evaluate TSP gain
            candidate_costs = get_tsp_costs(candidate_solutions)
            delta = candidate_costs - current_costs

            # 5. Accept/Reject (SA)
            prob = torch.exp(-delta / T).clamp(max=1.0)
            accept = (delta <= 0) | (torch.rand(B, device=device, generator=self.generator) < prob)

            current_solutions[accept] = candidate_solutions[accept]
            current_costs[accept] = candidate_costs[accept]

            improved = candidate_costs < best_costs
            best_solutions[improved] = candidate_solutions[improved]
            best_costs[improved] = candidate_costs[improved]

            # 6. Weights
            scores = torch.zeros(B, device=device)
            scores[improved] = 10.0
            scores[accept & ~improved] = 2.0
            avg_score = scores.mean().item()
            self.d_weights[d_idx] = 0.9 * self.d_weights[d_idx] + 0.1 * avg_score
            self.r_weights[r_idx] = 0.9 * self.r_weights[r_idx] + 0.1 * avg_score

            self._viz_record(
                iteration=i,
                d_idx=d_idx,
                r_idx=r_idx,
                best_cost=float(best_costs.min().item()),
                current_cost=float(current_costs.mean().item()),
                temperature=float(T.mean().item()),
                n_improved=int(improved.sum().item()),
                n_accepted=int(accept.sum().item()),
            )

            T *= cooling_rate

        # Final conversion to CVRP routes using optimal split
        best_routes, final_costs = vectorized_linear_split(
            best_solutions, self.dist_matrix, self.wastes, self.vehicle_capacity, max_vehicles=max_vehicles
        )

        return best_routes, final_costs

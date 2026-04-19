"""
Tabu Search (TS) for VRPP.

Classic metaheuristic using adaptive memory and responsive exploration.
Implements short-term memory (recency-based), long-term memory
(frequency-based), aspiration criteria, intensification, and diversification
as described in Glover's "Tabu Search Fundamentals and Uses" (1995).

Reference:
    Glover, F. "Tabu Search Fundamentals and Uses", 1995.
"""

import copy
import random
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import move_2opt_intra
from logic.src.policies.helpers.operators.intra_route_local_search.relocate import move_relocate
from logic.src.policies.helpers.operators.intra_route_local_search.swap import move_swap
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion,
    greedy_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes

from .params import TSParams


class TSSolver:
    """
    Tabu Search solver for VRPP with comprehensive memory structures.

    Key Features:
    - Short-term memory: Recency-based tabu list with dynamic tenure
    - Aspiration criteria: Override tabu status for globally improving moves
    - Long-term memory: Frequency-based memory for intensification/diversification
    - Elite solutions: Maintain pool of best solutions for path relinking
    - Strategic oscillation: Explore feasible/infeasible boundaries (optional)
    - Candidate lists: Restrict neighborhood exploration for efficiency
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: TSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Short-term memory: Tabu list stores (move_type, move_attributes, expiration_iter)
        # move_attributes is a tuple representing the move (e.g., (node1, node2) for swap)
        self.tabu_list: Deque[Tuple[str, Any, int]] = deque()

        # Long-term memory: Frequency matrices
        # Track how often nodes appear in solutions and how often moves are performed
        self.node_frequency: DefaultDict[int, int] = defaultdict(int)
        self.move_frequency: DefaultDict[Tuple[str, Any], int] = defaultdict(int)

        # Elite solutions pool for intensification and path relinking
        self.elite_solutions: List[Tuple[List[List[int]], float]] = []

        # Iteration counters
        self.iteration = 0
        self.iterations_since_improvement = 0

        # LLH pool for destroy-repair
        self._llh_pool = [
            self._llh_random_greedy,
            self._llh_worst_regret,
            self._llh_cluster_greedy,
            self._llh_worst_greedy,
            self._llh_random_regret,
        ]

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run Tabu Search with adaptive memory structures.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Initialize solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        current_routes = copy.deepcopy(routes)
        current_profit = profit
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Setup modular acceptance criterion
        self.params.acceptance_criterion.setup(profit)

        # Add initial solution to elite pool
        self._update_elite_pool(routes, profit)
        self._update_frequency_memory(routes)

        for iteration in range(self.params.max_iterations):
            self.iteration = iteration
            # Time limit check
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Restart if stuck
            if self.iterations_since_improvement > self.params.max_iterations_no_improve:
                if self.params.diversification_enabled:
                    current_routes = self._diversification_restart()
                    current_profit = self._evaluate(current_routes)
                    self.iterations_since_improvement = 0
                else:
                    break

            # Periodic intensification
            if (
                self.params.intensification_enabled
                and self.iteration > 0
                and self.iteration % self.params.intensification_interval == 0
            ):
                current_routes = self._intensification_phase(best_routes)
                current_profit = self._evaluate(current_routes)

            # Periodic diversification
            if (
                self.params.diversification_enabled
                and self.iteration > 0
                and self.iteration % self.params.diversification_interval == 0
            ):
                current_routes = self._diversification_phase(current_routes)
                current_profit = self._evaluate(current_routes)

            # Generate candidate neighborhood and select best move
            candidates = self._generate_candidates(current_routes)
            best_candidate, best_candidate_profit, best_move_desc = self._select_best_candidate(
                candidates, current_profit, best_profit
            )

            # Move to best candidate
            if best_candidate is not None:
                # Determine if this specific move was accepted (using the same logic as selection)
                # Note: selection already filtered by acceptance, but we need the flag for step()
                is_tabu = self._is_tabu(best_move_desc) if best_move_desc else False
                is_accepted, _ = self.params.acceptance_criterion.accept(
                    current_obj=current_profit,
                    candidate_obj=best_candidate_profit,
                    is_tabu=is_tabu,
                    best_profit=best_profit,
                    iteration=iteration,
                )

                current_routes = best_candidate
                current_profit = best_candidate_profit

                # Update tabu list
                if best_move_desc is not None:
                    self._add_to_tabu_list(best_move_desc)
                    self._update_move_frequency(best_move_desc)

                # Update frequency memory
                self._update_frequency_memory(current_routes)

                # Update best solution
                if current_profit > best_profit:
                    best_routes = copy.deepcopy(current_routes)
                    best_profit = current_profit
                    self.iterations_since_improvement = 0
                    self._update_elite_pool(current_routes, current_profit)
                else:
                    self.iterations_since_improvement += 1

                # Step criterion
                self.params.acceptance_criterion.step(
                    current_obj=current_profit,
                    candidate_obj=best_candidate_profit,
                    accepted=is_accepted,
                    iteration=iteration,
                )

            # Record telemetry
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=self.iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                tabu_size=len(self.tabu_list),
                elite_size=len(self.elite_solutions),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ========================================================================
    # Candidate Selection
    # ========================================================================

    def _select_best_candidate(
        self,
        candidates: Sequence[Tuple[List[List[int]], Tuple[str, Any]]],
        current_profit: float,
        best_profit: float,
    ) -> Tuple[Optional[List[List[int]]], float, Optional[Tuple[str, Any]]]:
        """
        Select best accepted candidate based on modular criterion (non-tabu or aspirated).
        """
        best_candidate = None
        best_candidate_profit = float("-inf")
        best_move_desc = None

        for candidate_routes, move_desc in candidates:
            candidate_profit = self._evaluate(candidate_routes)
            is_tabu = self._is_tabu(move_desc)

            # Accept if not tabu or if aspiration criteria met
            # Delegate decision to injected criterion
            is_accepted, _ = self.params.acceptance_criterion.accept(
                current_obj=current_profit,
                candidate_obj=candidate_profit,
                is_tabu=is_tabu,
                best_profit=best_profit,
                iteration=self.iteration,
            )

            if not is_accepted:
                continue

            # Apply frequency-based penalty for diversification (Objective selection phase)
            penalty = 0.0
            if self.params.diversification_enabled:
                penalty = self._compute_frequency_penalty(move_desc)

            adjusted_profit = candidate_profit - penalty

            if adjusted_profit > best_candidate_profit:
                best_candidate = candidate_routes
                best_candidate_profit = candidate_profit
                best_move_desc = move_desc

        # If all moves are rejected, force the least-rejected one (Tabu search convention)
        if best_candidate is None and candidates:
            best_candidate, best_move_desc = candidates[0]
            best_candidate_profit = self._evaluate(best_candidate)

        return best_candidate, best_candidate_profit, best_move_desc

    # ========================================================================
    # Tabu List Management (Short-term Memory)
    # ========================================================================

    def _is_tabu(self, move_desc: Tuple[str, Any]) -> bool:
        """Check if a move is currently tabu."""
        move_type, move_attrs = move_desc
        for tabu_type, tabu_attrs, expiration in self.tabu_list:
            if tabu_type == move_type and tabu_attrs == move_attrs and expiration > self.iteration:
                return True
        return False

    def _add_to_tabu_list(self, move_desc: Tuple[str, Any]):
        """Add a move to the tabu list with appropriate tenure."""
        move_type, move_attrs = move_desc

        # Compute tabu tenure
        tenure = self._compute_dynamic_tenure() if self.params.dynamic_tenure else self.params.tabu_tenure

        expiration = self.iteration + tenure
        self.tabu_list.append((move_type, move_attrs, expiration))

        # Clean expired entries
        self._clean_tabu_list()

    def _clean_tabu_list(self):
        """Remove expired tabu entries."""
        while self.tabu_list and self.tabu_List[0][2] <= self.iteration:
            self.tabu_list.popleft()

    def _compute_dynamic_tenure(self) -> int:
        """
        Compute dynamic tabu tenure based on search state.

        Shorter tenure during improvement, longer during stagnation.
        """
        base_tenure = self.params.tabu_tenure

        if self.iterations_since_improvement < 10:
            # Rapid improvement — shorter tenure for exploitation
            return max(self.params.min_tenure, int(base_tenure * 0.7))
        elif self.iterations_since_improvement > 50:
            # Stagnation — longer tenure for diversification
            return min(self.params.max_tenure, int(base_tenure * 1.5))
        else:
            return base_tenure

    # ========================================================================
    # Frequency Memory (Long-term Memory)
    # ========================================================================

    def _update_frequency_memory(self, routes: List[List[int]]):
        """Update node frequency counters."""
        for route in routes:
            for node in route:
                self.node_frequency[node] += 1

    def _update_move_frequency(self, move_desc: Tuple[str, Any]):
        """Update move frequency counters."""
        self.move_frequency[move_desc] += 1

    def _compute_frequency_penalty(self, move_desc: Tuple[str, Any]) -> float:
        """
        Compute penalty based on move frequency to encourage diversification.

        Frequently used moves receive higher penalties.
        """
        freq = self.move_frequency.get(move_desc, 0)
        return self.params.frequency_penalty_weight * freq

    # ========================================================================
    # Elite Solutions & Intensification
    # ========================================================================

    def _update_elite_pool(self, routes: List[List[int]], profit: float):
        """Maintain a pool of elite (high-quality) solutions."""
        # Add to elite pool
        self.elite_solutions.append((copy.deepcopy(routes), profit))

        # Sort by profit (descending)
        self.elite_solutions.sort(key=lambda x: x[1], reverse=True)

        # Keep only top-k elite solutions
        self.elite_solutions = self.elite_solutions[: self.params.elite_size]

    def _intensification_phase(self, best_routes: List[List[int]]) -> List[List[int]]:
        """
        Intensification: Return to best solution and explore neighborhood intensively.

        This encourages exploitation of promising regions.
        """
        # Start from best known solution
        routes = copy.deepcopy(best_routes)

        # Apply local search intensively
        for _ in range(5):
            improved = False
            for llh in self._llh_pool:
                try:
                    new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                    new_profit = self._evaluate(new_routes)
                    if new_profit > self._evaluate(routes):
                        routes = new_routes
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                break

        return routes

    # ========================================================================
    # Diversification
    # ========================================================================

    def _diversification_phase(self, current_routes: List[List[int]]) -> List[List[int]]:
        """
        Diversification: Perturb current solution to explore new regions.

        Uses frequency-based penalties to favor rarely visited nodes.
        """
        routes = copy.deepcopy(current_routes)

        # Remove frequently visited nodes
        all_nodes_in_routes = [node for route in routes for node in route]
        if not all_nodes_in_routes:
            return routes

        # Sort by frequency (descending)
        freq_nodes = sorted(all_nodes_in_routes, key=lambda n: self.node_frequency.get(n, 0), reverse=True)

        # Remove top frequent nodes
        n_remove = min(self.params.n_removal * 2, len(freq_nodes) // 2)
        nodes_to_remove = set(freq_nodes[:n_remove])

        # Rebuild routes without these nodes
        new_routes = []
        removed_nodes = []
        for route in routes:
            new_route = [n for n in route if n not in nodes_to_remove]
            removed_nodes.extend([n for n in route if n in nodes_to_remove])
            if new_route:
                new_routes.append(new_route)

        # Reinsert removed nodes
        if removed_nodes:
            if self.params.profit_aware_operators:
                new_routes = greedy_profit_insertion(
                    new_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                new_routes = greedy_insertion(
                    new_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )

        return new_routes

    def _diversification_restart(self) -> List[List[int]]:
        """
        Diversification restart: Build a new solution avoiding frequent patterns.

        This is triggered when search stagnates.
        """
        # If we have elite solutions, use path relinking
        if len(self.elite_solutions) >= 2:
            return self._path_relink(self.elite_solutions[0][0], self.elite_solutions[1][0])

        # Otherwise, build a new solution with frequency penalties
        return self._build_diversified_solution()

    def _build_diversified_solution(self) -> List[List[int]]:
        """
        Build a new solution that avoids frequently used nodes.

        Penalize nodes with high frequency to encourage exploration.
        """
        # Modify wastes temporarily to penalize frequent nodes
        modified_wastes = {}
        max_freq = max(self.node_frequency.values()) if self.node_frequency else 1

        for node in self.nodes:
            freq = self.node_frequency.get(node, 0)
            penalty = (freq / max_freq) * 0.3  # Up to 30% penalty
            modified_wastes[node] = self.wastes.get(node, 0.0) * (1.0 - penalty)

        # Build solution with modified wastes
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=modified_wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _path_relink(self, routes1: List[List[int]], routes2: List[List[int]]) -> List[List[int]]:
        """
        Path relinking: Generate intermediate solution between two elite solutions.

        This explores trajectories between high-quality solutions.
        """
        # Simple implementation: blend node selections
        nodes1 = set(node for route in routes1 for node in route)
        nodes2 = set(node for route in routes2 for node in route)

        # Nodes in both solutions are kept
        common_nodes = nodes1 & nodes2

        # Randomly select from differing nodes
        diff_nodes = (nodes1 | nodes2) - common_nodes
        selected_diff = set(self.random.sample(list(diff_nodes), min(len(diff_nodes), len(diff_nodes) // 2)))

        # Build routes with selected nodes
        selected_nodes = list(common_nodes | selected_diff)

        # Use greedy insertion to rebuild
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            return greedy_insertion(
                [],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    # ========================================================================
    # Neighborhood Generation
    # ========================================================================

    def _generate_candidates(self, routes: List[List[int]]) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """
        Generate candidate neighborhood moves.

        Returns list of (new_routes, move_description) tuples.
        """
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        # Use candidate list if enabled
        max_candidates: float = self.params.candidate_list_size if self.params.candidate_list_enabled else float("inf")

        # Destroy-repair based neighborhoods
        if self.params.use_insertion:
            for _ in range(min(3, int(max_candidates))):
                llh_idx = self.random.randint(0, len(self._llh_pool) - 1)
                try:
                    new_routes = self._llh_pool[llh_idx](copy.deepcopy(routes), self.params.n_removal)
                    move_desc = ("destroy_repair", (llh_idx, self.iteration))
                    candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        # Swap-based neighborhoods
        if self.params.use_swap and len(candidates) < max_candidates:
            candidates.extend(self._generate_swap_moves(routes, max_new=3))

        # Relocate-based neighborhoods
        if self.params.use_relocate and len(candidates) < max_candidates:
            candidates.extend(self._generate_relocate_moves(routes, max_new=3))

        # 2-opt neighborhoods
        if self.params.use_2opt and len(candidates) < max_candidates:
            candidates.extend(self._generate_2opt_moves(routes, max_new=2))

        # Shuffle to avoid bias
        self.random.shuffle(candidates)

        return candidates[: int(max_candidates)]

    def _generate_swap_moves(
        self, routes: List[List[int]], max_new: int = 5
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """Generate swap-based neighborhood moves."""
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            # Try inter-route swap
            if len(new_routes) >= 2:
                try:
                    r1_idx = self.random.randint(0, len(new_routes) - 1)
                    r2_idx = self.random.randint(0, len(new_routes) - 1)
                    if r1_idx != r2_idx and new_routes[r1_idx] and new_routes[r2_idx]:
                        pos1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r2_idx]) - 1)
                        node1 = new_routes[r1_idx][pos1]
                        node2 = new_routes[r2_idx][pos2]

                        adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                        move_swap(adapter, node1, node2, r1_idx, pos1, r2_idx, pos2)
                        new_routes = adapter.routes
                        move_desc = ("swap_inter", (node1, node2))
                        candidates.append((new_routes, move_desc))
                except Exception:
                    continue

            # Try intra-route swap
            if not candidates or self.random.random() < 0.5:
                try:
                    r_idx = self.random.randint(0, len(new_routes) - 1)
                    if len(new_routes[r_idx]) >= 2:
                        pos1 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        if pos1 != pos2:
                            node1 = new_routes[r_idx][pos1]
                            node2 = new_routes[r_idx][pos2]
                            adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                            move_swap(adapter, node1, node2, r_idx, pos1, r_idx, pos2)
                            new_routes = adapter.routes
                            move_desc = ("swap_intra", (node1, node2))
                            candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        return candidates

    def _generate_relocate_moves(
        self, routes: List[List[int]], max_new: int = 5
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """Generate relocate-based neighborhood moves."""
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            # Try inter-route relocate
            if len(new_routes) >= 2:
                try:
                    r1_idx = self.random.randint(0, len(new_routes) - 1)
                    r2_idx = self.random.randint(0, len(new_routes) - 1)
                    if r1_idx != r2_idx and new_routes[r1_idx]:
                        pos1 = self.random.randint(0, len(new_routes[r1_idx]) - 1)
                        node = new_routes[r1_idx][pos1]
                        pos2 = self.random.randint(0, len(new_routes[r2_idx]))

                        adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                        if pos2 == 0:
                            v, p_v = 0, -1
                        else:
                            v, p_v = new_routes[r2_idx][pos2 - 1], pos2 - 1
                        move_relocate(adapter, node, v, r1_idx, pos1, r2_idx, p_v)
                        new_routes = adapter.routes

                        move_desc = ("relocate_inter", (node, r1_idx, r2_idx))
                        candidates.append((new_routes, move_desc))
                except Exception:
                    continue

            # Try intra-route relocate
            if not candidates or self.random.random() < 0.5:
                try:
                    r_idx = self.random.randint(0, len(new_routes) - 1)
                    if len(new_routes[r_idx]) >= 2:
                        pos1 = self.random.randint(0, len(new_routes[r_idx]) - 1)
                        pos2 = self.random.randint(0, len(new_routes[r_idx]))
                        if pos1 != pos2:
                            node = new_routes[r_idx][pos1]
                            adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                            if pos2 == 0:
                                v, p_v = 0, -1
                            else:
                                v, p_v = new_routes[r_idx][pos2 - 1], pos2 - 1
                            move_relocate(adapter, node, v, r_idx, pos1, r_idx, p_v)
                            new_routes = adapter.routes

                            move_desc = ("relocate_intra", (node, pos1, pos2))
                            candidates.append((new_routes, move_desc))
                except Exception:
                    continue

        return candidates

    def _generate_2opt_moves(
        self, routes: List[List[int]], max_new: int = 3
    ) -> Sequence[Tuple[List[List[int]], Tuple[str, Any]]]:
        """Generate 2-opt neighborhood moves."""
        candidates: List[Tuple[List[List[int]], Tuple[str, Any]]] = []

        for _ in range(max_new):
            new_routes = copy.deepcopy(routes)

            try:
                r_idx = self.random.randint(0, len(new_routes) - 1)
                if len(new_routes[r_idx]) >= 4:
                    # For 2-opt, we need two edges (u, next_u) and (v, next_v)
                    # We'll pick random pos1, pos2
                    pos1 = self.random.randint(0, len(new_routes[r_idx]) - 2)
                    pos2 = self.random.randint(pos1 + 2, len(new_routes[r_idx]) - 1)
                    u = new_routes[r_idx][pos1]
                    v = new_routes[r_idx][pos2]

                    adapter = TSLSAdapter(new_routes, self.dist_matrix, self.wastes, self.capacity, -1e9)
                    move_2opt_intra(adapter, u, v, r_idx, pos1, r_idx, pos2)
                    new_routes = adapter.routes

                    move_desc = ("2opt", (r_idx, self.iteration))
                    candidates.append((new_routes, move_desc))
            except Exception:
                continue

        return candidates

    # ========================================================================
    # LLH Pool (Destroy-Repair Heuristics)
    # ========================================================================

    def _llh_random_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            # random_profit_removal reverted to random_removal
            partial, removed = random_removal(routes, n, rng=self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            partial, removed = random_removal(routes, n, rng=self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh_worst_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh_cluster_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            # cluster_profit_removal reverted to cluster_removal
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh_worst_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh_random_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            # random_profit_removal reverted to random_removal
            partial, removed = random_removal(routes, n, rng=self.random)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            partial, removed = random_removal(routes, n, rng=self.random)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _build_initial_solution(self) -> List[List[int]]:
        """Build initial solution using greedy profit-aware heuristic."""
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate solution profit (revenue - cost)."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Compute total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total


class TSLSAdapter:
    """Adapter for HGS-style operators to work with TSSolver."""

    def __init__(self, routes, dist_matrix, wastes, capacity, cost_unit):
        self.routes = routes
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.C = cost_unit
        self.params = type("Params", (), {"use_relocate_chain": False})()
        self._load_cache = {}

    def _get_load_cached(self, r_idx: int) -> float:
        if r_idx not in self._load_cache:
            self._load_cache[r_idx] = sum(self.waste.get(n, 0) for n in self.routes[r_idx])
        return self._load_cache[r_idx]

    def _update_map(self, affected_indices: Set[int]):
        for r_idx in affected_indices:
            if r_idx in self._load_cache:
                del self._load_cache[r_idx]

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(n, 0) for n in r)

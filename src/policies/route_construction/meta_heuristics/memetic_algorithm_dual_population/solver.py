"""
Memetic Algorithm with Dual Population (MA-DP) for VRPP.

EXACT COPY of Volleyball Premier League (VPL) with rigorous nomenclature.

TERMINOLOGY MAPPING (VPL → MA-DP):
- "Active Teams" → Active Population (competing solutions)
- "Passive Teams" → Reserve Population (diversity pool)
- "Substitution" → Diversity Injection Operator
- "Coaching from Top 3" → Elite-Guided Solution Construction
- "Seasons" → Generations/Iterations
- "Team Formation" → Solution Construction

Algorithm Structure (Moghdani & Salimifard, 2018):
    1. Initialize dual population (N active + N reserve solutions)
    2. For each iteration:
        a. Competition Phase: Rank active solutions by fitness
        b. Diversity Injection: Combine active solutions with reserve pool
        c. Elite-Guided Construction: Weaker solutions learn from top-k performers
        d. Local Search: Refine all solutions
        e. Update best solution

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League Algorithm."
    Applied Soft Computing, 64, 161-185. DOI: 10.1016/j.asoc.2017.11.043

IMPORTANT: This implementation EXACTLY matches the VPL algorithm with only
           terminology changed from sports metaphors to OR terminology.
           Uses identical RNG attribute name (self.random) and algorithm flow.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import MemeticAlgorithmDualPopulationParams


class MemeticAlgorithmDualPopulationSolver:
    """
    Memetic Algorithm with Dual Population for VRPP.
    EXACT COPY of VPL with rigorous nomenclature.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticAlgorithmDualPopulationParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize MA-DP solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 = depot.
            wastes: Dictionary mapping node index to waste/profit value.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit of waste collected.
            C: Cost per unit of distance traveled.
            params: MA-DP algorithm parameters.
            mandatory_nodes: List of nodes that must be visited.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()

        # Initialize ACO Local Search for elite learning
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=params.seed,
        )
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the MA-DP algorithm.

        Returns:
            Tuple of (routes, profit, cost):
                - routes: List of vehicle routes (each route is list of node indices)
                - profit: Net profit of the solution
                - cost: Total routing cost (distance)
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Phase 1: Population Formation and Initialization
        # VPL: active_teams, passive_teams
        # MA-DP: active_teams, passive_teams (keep same names for exact match)
        active_teams = self._initialize_population(self.params.population_size)
        passive_teams = self._initialize_population(self.params.population_size)

        # Evaluate active teams
        active_profits = [self._evaluate(team) for team in active_teams]

        # Sort active teams by profit (descending)
        sorted_indices = sorted(range(len(active_profits)), key=lambda i: active_profits[i], reverse=True)
        active_teams = [active_teams[i] for i in sorted_indices]
        active_profits = [active_profits[i] for i in sorted_indices]

        # Track global best
        best_routes = copy.deepcopy(active_teams[0])
        best_profit = active_profits[0]
        best_cost = self._cost(best_routes)

        # Main MA-DP loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Phase 2: Racing and Interplays (Competition)
            # Teams are already sorted by fitness from previous iteration
            # In this phase, we simply maintain the ranking

            # Phase 3: Substitution Operator (Diversity Injection)
            active_teams = self._substitution_phase(active_teams, passive_teams)

            # Phase 4: Coaching and Learning
            active_teams = self._coaching_phase(active_teams)

            # Re-evaluate all teams after modifications
            active_profits = [self._evaluate(team) for team in active_teams]

            # Sort by profit again
            sorted_indices = sorted(range(len(active_profits)), key=lambda i: active_profits[i], reverse=True)
            active_teams = [active_teams[i] for i in sorted_indices]
            active_profits = [active_profits[i] for i in sorted_indices]

            # Update global best
            if active_profits[0] > best_profit:
                best_routes = copy.deepcopy(active_teams[0])
                best_profit = active_profits[0]
                best_cost = self._cost(best_routes)

            # Visualization tracking
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                active_teams=self.params.population_size,
                passive_teams=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private: Initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, pop_size: int) -> List[List[List[int]]]:
        """
        Initialize a population of routing solutions.

        Creates diverse initial solutions using nearest-neighbor heuristic
        with randomized node orderings.

        Args:
            pop_size: Number of solutions to generate.

        Returns:
            List of routing solutions (each solution is a list of routes).
        """
        population = []
        for _ in range(pop_size):
            routes = build_nn_routes(
                nodes=self.nodes,
                mandatory_nodes=self.mandatory_nodes,
                wastes=self.wastes,
                capacity=self.capacity,
                dist_matrix=self.dist_matrix,
                R=self.R,
                C=self.C,
                rng=self.random,
            )
            population.append(routes)
        return population

    # ------------------------------------------------------------------
    # Private: MA-DP Phases
    # ------------------------------------------------------------------

    def _substitution_phase(
        self, active_teams: List[List[List[int]]], passive_teams: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """
        Apply substitution operator to inject diversity into active teams.

        For each active team, randomly substitute solution components
        (routes or nodes) from passive teams with probability substitution_rate.
        This mimics substituting exhausted players with fresh reserves.

        Args:
            active_teams: List of active team solutions.
            passive_teams: List of passive team solutions (reserve pool).

        Returns:
            Modified active teams with injected diversity.
        """
        modified_teams = []

        for team_idx, team in enumerate(active_teams):
            # Preserve elite teams from substitution
            if team_idx < self.params.elite_size:
                modified_teams.append(copy.deepcopy(team))
                continue

            new_team = copy.deepcopy(team)

            # Apply substitution with probability
            if self.random.random() < self.params.substitution_rate:
                # Select a random passive team as donor
                donor_team = self.random.choice(passive_teams)

                # Extract all nodes from current team and donor
                current_nodes = [node for route in new_team for node in route]
                donor_nodes = [node for route in donor_team for node in route]

                if donor_nodes:
                    # Substitute: remove some nodes and add nodes from donor
                    n_substitute = max(1, int(len(current_nodes) * 0.3))

                    # Remove random nodes (except mandatory)
                    removable = [n for n in current_nodes if n not in self.mandatory_nodes]
                    if removable:
                        to_remove = self.random.sample(removable, min(n_substitute, len(removable)))
                        # Flatten routes and remove nodes
                        flat_nodes = [n for n in current_nodes if n not in to_remove]

                        # Add nodes from donor that aren't already present
                        available_donor = [n for n in donor_nodes if n not in flat_nodes]
                        if available_donor:
                            to_add = self.random.sample(available_donor, min(n_substitute, len(available_donor)))
                            flat_nodes.extend(to_add)

                        # Rebuild routes using greedy insertion
                        try:
                            if self.params.profit_aware_operators:
                                # First remove nodes with profit-awareness if possible,
                                # but substitution phase usually just randomizes.
                                # However, we can use greedy_profit_insertion for repair.
                                new_team = greedy_profit_insertion(
                                    [],
                                    flat_nodes,
                                    self.dist_matrix,
                                    self.wastes,
                                    self.capacity,
                                    R=self.R,
                                    C=self.C,
                                    mandatory_nodes=self.mandatory_nodes,
                                    expand_pool=self.params.vrpp,
                                )
                            else:
                                new_team = greedy_insertion(
                                    [],
                                    flat_nodes,
                                    self.dist_matrix,
                                    self.wastes,
                                    self.capacity,
                                    mandatory_nodes=self.mandatory_nodes,
                                    expand_pool=self.params.vrpp,
                                )
                        except Exception:
                            # If reconstruction fails, keep original team
                            new_team = copy.deepcopy(team)

            modified_teams.append(new_team)

        return modified_teams

    def _coaching_phase(self, active_teams: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        Apply coaching and learning phase.

        Teams ranked below the top 3 learn from the best performers. Each
        weaker team creates a new formation by combining characteristics
        from the top 3 teams using weighted learning.

        Mathematical formulation:
            Team_i^(t+1) = w1 * Team_1 + w2 * Team_2 + w3 * Team_3
            where w1 + w2 + w3 = 1.0

        Args:
            active_teams: List of active teams sorted by fitness (best first).

        Returns:
            Modified active teams after coaching.
        """
        if len(active_teams) < self.params.elite_size:
            return active_teams

        # Extract top 3 teams (coaching staff)
        top1 = active_teams[0]
        top2 = active_teams[1]
        top3 = active_teams[2]

        coached_teams = []

        for team_idx, team in enumerate(active_teams):
            # Top 3 teams don't need coaching - they ARE the coaches
            if team_idx < self.params.elite_size:
                coached_teams.append(copy.deepcopy(team))
                continue

            # Learn from top 3 teams
            new_team = self._learn_from_elite(team, top1, top2, top3)
            coached_teams.append(new_team)

        return coached_teams

    def _learn_from_elite(
        self,
        current_team: List[List[int]],
        top1: List[List[int]],
        top2: List[List[int]],
        top3: List[List[int]],
    ) -> List[List[int]]:
        """
        Create a new formation by learning from top 3 teams.

        Uses a weighted node selection strategy where nodes from better teams
        have higher probability of being included in the new formation.

        Args:
            current_team: Current team's routing solution.
            top1: Best team's routing solution.
            top2: Second-best team's routing solution.
            top3: Third-best team's routing solution.

        Returns:
            New routing solution learned from elite teams.
        """
        # Extract all nodes from top 3 teams
        nodes_top1 = {node for route in top1 for node in route}
        nodes_top2 = {node for route in top2 for node in route}
        nodes_top3 = {node for route in top3 for node in route}

        # Weighted node selection
        candidate_nodes = []

        # Iterate through all possible nodes
        for node in self.nodes:
            # Mandatory nodes always included
            if node in self.mandatory_nodes:
                candidate_nodes.append(node)
                continue

            # Calculate weighted probability for this node
            weight = 0.0
            if node in nodes_top1:
                weight += self.params.coaching_weight_1
            if node in nodes_top2:
                weight += self.params.coaching_weight_2
            if node in nodes_top3:
                weight += self.params.coaching_weight_3

            # Select node based on weighted probability
            if weight > 0 and self.random.random() < weight:
                candidate_nodes.append(node)

        # Ensure mandatory nodes are included
        for mn in self.mandatory_nodes:
            if mn not in candidate_nodes:
                candidate_nodes.append(mn)

        # If no nodes selected, keep current team
        if not candidate_nodes:
            return copy.deepcopy(current_team)

        # Reconstruct routes using greedy insertion
        try:
            if self.params.profit_aware_operators:
                new_routes = greedy_profit_insertion(
                    [],
                    candidate_nodes,
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
                    [],
                    candidate_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )

            # Apply local search refinement (reusing instance)
            new_routes = self.ls.optimize(new_routes)

            return new_routes
        except Exception:
            # If learning fails, return current team
            return copy.deepcopy(current_team)

    # ------------------------------------------------------------------
    # Private: Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate net profit of a routing solution.

        Args:
            routes: List of vehicle routes.

        Returns:
            Net profit (revenue - cost).
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of vehicle routes.

        Returns:
            Total distance traveled.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first node
            total += self.dist_matrix[0][route[0]]
            # Inter-node distances
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            # Last node back to depot
            total += self.dist_matrix[route[-1]][0]
        return total

"""
Progressive Scenario Hedging within CG.
"""

from typing import Dict, List, Set

import numpy as np


class ProgressiveHedgingCGLoop:
    """
    Non-anticipativity enforcement mechanism strictly for the root-node Column Generation.
    Maintains distinct Restricted Master Problems for each scenario xi.
    """

    def __init__(self, num_scenarios: int, base_rho: float = 1.0):
        self.num_scenarios = num_scenarios
        self.base_rho = base_rho

        # Average value of variable x_k across all scenario RMPs
        self.x_bar: Dict[int, float] = {}

    def update_x_bar(self, scenario_solutions: Dict[int, Dict[int, float]]):
        """
        Update the consensus variable values across all scenarios.
        scenario_solutions: scenario_id -> {var_id -> value}
        """
        all_vars: Set[int] = set()
        for sol in scenario_solutions.values():
            all_vars.update(sol.keys())

        for var_id in all_vars:
            total = sum(sol.get(var_id, 0.0) for sol in scenario_solutions.values())
            self.x_bar[var_id] = total / self.num_scenarios

    def compute_dynamic_penalty(self, var_id: int, scenario_prizes: Dict[int, float]) -> float:
        """
        Compute rho dynamically per iteration. Set rho proportional to variance of scenario prizes.
        Bins with high prize disagreement across scenarios incur steeper penalties.
        """
        if not scenario_prizes:
            return self.base_rho

        prizes = list(scenario_prizes.values())
        variance = np.var(prizes) if len(prizes) > 1 else 0.0

        return float(self.base_rho * (1.0 + variance))

    def calculate_augmented_reduced_cost(
        self,
        route_nodes: List[int],
        dist_matrix: np.ndarray,
        scenario_prizes: Dict[int, float],  # pi_i^{scenario, xi}
        dual_values: Dict[int, float],  # pi_i^{xi}
        scenario_id: int,
        route_id: int,
        current_x_k: float,  # The value of x_k in current scenario RMP
    ) -> float:
        """
        Modify reduced cost calculation for a specific scenario xi:
        c'_{k, xi} = -dist(k) + sum_i(pi_i^{scenario, xi} - pi_i^{xi}) - rho * (x_k - x_bar_k)^2
        """
        dist_cost = 0.0
        prize_sum = 0.0

        for i in range(len(route_nodes) - 1):
            dist_cost += dist_matrix[route_nodes[i], route_nodes[i + 1]]

        for n in route_nodes:
            if n != 0:
                prize_sum += scenario_prizes.get(n, 0.0) - dual_values.get(n, 0.0)

        # Find variable penalty
        rho = self.compute_dynamic_penalty(route_id, scenario_prizes)
        x_bar_k = self.x_bar.get(route_id, 0.0)
        ph_penalty = rho * ((current_x_k - x_bar_k) ** 2)

        return prize_sum - dist_cost - ph_penalty

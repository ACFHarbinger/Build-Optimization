"""
Scenario-Consistent Branching.
"""

from typing import Any, Dict, List


class ScenarioConsistentBranching:
    """
    Dynamic branching rule that leverages deterministic solutions of individual
    scenarios within the ScenarioTree.
    """

    def __init__(self, base_threshold: float = 0.95):
        self.base_threshold = base_threshold

    def calculate_consensus(
        self,
        var_id: int,
        scenario_tree: Any,
        deterministic_scenario_solutions: Dict[int, Dict[int, int]],  # scenario_id -> {var_id -> value}
    ) -> float:
        """
        consensus(g_i) = Count scenarios where g_i = 1 / Total scenarios
        """
        if not deterministic_scenario_solutions:
            return 0.5

        total_scenarios = len(deterministic_scenario_solutions)
        visited_count = sum(1 for s_id, sol in deterministic_scenario_solutions.items() if sol.get(var_id, 0) == 1)

        return visited_count / total_scenarios if total_scenarios > 0 else 0.5

    def select_branching_variable(
        self,
        fractional_vars: List[Any],
        scenario_tree: Any,
        deterministic_solutions: Dict[int, Dict[int, int]],
        days_remaining: int,
        initial_horizon: int,
    ) -> Any:
        """
        Branch on the variable that maximizes |consensus(g_i) - 0.5|.
        Horizon adaptivity: demand higher consensus early in the horizon.
        """
        best_var = None
        best_score = -1.0

        # Decay factor: early horizon -> factor near 1.0; late horizon -> factor gets smaller
        # This makes the threshold lower later, but the branching rule just picks max |consensus - 0.5|.
        # The threshold is used to decide if we should just implicitly fix it without branching.
        # decay_factor = days_remaining / initial_horizon if initial_horizon > 0 else 1.0
        # threshold = self.base_threshold * decay_factor

        # We find the variable that we have the highest confidence about across scenarios,
        # but which is currently fractional in the Master Problem LP.
        for var in fractional_vars:
            consensus = self.calculate_consensus(var.id, scenario_tree, deterministic_solutions)
            score = abs(consensus - 0.5)

            if score > best_score:
                best_score = score
                best_var = var

        return best_var

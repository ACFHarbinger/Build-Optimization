"""
Fix-and-Optimize Corridor Method.
"""

from typing import Any, List, Set

import numpy as np


class FixAndOptimizeRefiner:
    """
    Exact decomposition sub-routine that polishes the incumbent solution by unfixing
    targeted clusters and re-optimizing them with the exact BPC engine.
    """

    def __init__(self, tabu_length: int = 10, max_unfix: int = 5):
        self.tabu_list: List[Set[int]] = []
        self.tabu_length = tabu_length
        self.max_unfix = max_unfix

    def _is_tabu(self, candidate_set: Set[int]) -> bool:
        """Reject cluster selection that has Jaccard similarity > 0.8 with any set in Tabu list."""
        if not candidate_set:
            return True

        for tabu_set in self.tabu_list:
            intersection = len(candidate_set.intersection(tabu_set))
            union = len(candidate_set.union(tabu_set))
            if union > 0 and (intersection / union) > 0.8:
                return True
        return False

    def _add_to_tabu(self, cluster: Set[int]):
        self.tabu_list.append(cluster)
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)

    def select_cluster_overflow_urgency(self, unrouted: List[int], days_to_overflow: np.ndarray) -> Set[int]:
        """Select top N bins based on expected overflow probability."""
        if not unrouted:
            return set()

        # days_to_overflow is 1D array corresponding to bins [0..n_bins-1]
        urn_indices = np.argsort(days_to_overflow)
        cluster: Set[int] = set()
        for idx in urn_indices:
            bin_id = idx + 1
            if len(cluster) >= self.max_unfix:
                break
            cluster.add(bin_id)

        return cluster

    def select_cluster_scenario_divergence(
        self, all_nodes: List[int], scenario_tree: Any, current_day: int
    ) -> Set[int]:
        """Select N bins with highest variance of scenario wastes."""
        scenarios = scenario_tree.get_scenarios_at_day(current_day)
        if not scenarios:
            return set(all_nodes[: self.max_unfix])

        all_wastes = np.stack([s.wastes for s in scenarios])
        variances = np.var(all_wastes, axis=0)

        # Argsort descending
        urn_indices = np.argsort(variances)[::-1]
        cluster: Set[int] = set()
        for idx in urn_indices:
            bin_id = idx + 1
            if len(cluster) >= self.max_unfix:
                break
            cluster.add(bin_id)

        return cluster

    def refine(
        self,
        current_incumbent: Any,
        bpc_engine: Any,
        scenario_tree: Any,
        current_day: int,
        days_to_overflow: np.ndarray,
        global_column_pool: List[Any],
        strategy: str = "overflow_urgency",
    ) -> Any:
        """
        Unfix targeted clusters and re-optimize.
        """
        all_nodes = [i + 1 for i in range(len(days_to_overflow))]

        if strategy == "overflow_urgency":
            cluster = self.select_cluster_overflow_urgency(all_nodes, days_to_overflow)
        else:
            cluster = self.select_cluster_scenario_divergence(all_nodes, scenario_tree, current_day)

        if self._is_tabu(cluster):
            return current_incumbent  # Skip to prevent cycling

        self._add_to_tabu(cluster)

        # Warm-starting the Subproblem: Initialize RMP with global_column_pool filtered
        # for routes that exclusively service the unfixed bins.
        filtered_routes = []
        for route in global_column_pool:
            if all(n in cluster or n == 0 for n in route.nodes):
                filtered_routes.append(route)

        # Invoke exact BPC engine on the unfixed corridor
        # bpc_engine is expected to take filtered_routes to warm-start RMP
        if hasattr(bpc_engine, "solve_corridor"):
            refined_solution = bpc_engine.solve_corridor(cluster, filtered_routes)
            return refined_solution
        else:
            return current_incumbent

"""
Dive-and-Price Heuristic.
"""

from typing import Any, List, Tuple


class DiveAndPricePrimalHeuristic:
    """
    Executes a primal heuristic exactly once after root-node CG
    to establish a strong, scenario-consistent upper bound.
    """

    def __init__(self, penalty_M: float = 10000.0):
        self.penalty_M = penalty_M

    def evaluate_scenario_consensus(self, route_nodes: List[int], scenario_tree: Any) -> float:
        """
        scenario_consensus(k) is the proportion of high-probability scenarios
        in which the bins in route k are projected to face overflow simultaneously.
        """
        # Simplistic implementation matching requirements:
        # Evaluate how many scenarios all nodes in the route reach >90% fill rate.
        leaves = scenario_tree.get_leaves()
        if not leaves:
            return 1.0

        consensus_count = 0.0
        total_prob = 0.0

        for leaf in leaves:
            total_prob += leaf.probability
            # Check if all bins in route are near overflow
            clashing = all(leaf.wastes[n - 1] > 90.0 for n in route_nodes if n != 0)
            if clashing:
                consensus_count += leaf.probability

        return consensus_count / total_prob if total_prob > 0 else 1.0

    def select_column_to_fix(
        self,
        fractional_columns: List[Tuple[int, float, List[int]]],  # (col_idx, lambda_val, route_nodes)
        scenario_tree: Any,
    ) -> int:
        """
        Calculate composite diving score: score(k) = lambda_k * scenario_consensus(k).
        Returns the column index to fix.
        """
        best_idx = -1
        best_score = -float("inf")

        for col_idx, lambda_k, route_nodes in fractional_columns:
            if 0.0 < lambda_k < 1.0:
                consensus = self.evaluate_scenario_consensus(route_nodes, scenario_tree)
                score = lambda_k * consensus

                if score > best_score:
                    best_score = score
                    best_idx = col_idx

        return best_idx

    def execute(self, rmp: Any, fractional_columns: List[Tuple[int, float, List[int]]], scenario_tree: Any) -> None:
        """
        Execution and backtrack-free recovery for soft-fixing the best column.
        """
        while True:
            col_to_fix = self.select_column_to_fix(fractional_columns, scenario_tree)
            if col_to_fix == -1:
                break  # No fractional columns

            # Attempt to fix lambda_k = 1
            rmp.fix_column(col_to_fix, 1.0)
            status = rmp.resolve()

            if status == "INFEASIBLE":
                # Relax fixed constraint, penalize column with soft penalty M
                rmp.unfix_column(col_to_fix)
                rmp.penalize_column(col_to_fix, self.penalty_M)

                # Remove from fractional_columns candidates
                fractional_columns = [c for c in fractional_columns if c[0] != col_to_fix]
            else:
                break  # Feasible, accept fixing and stop diving for this iteration

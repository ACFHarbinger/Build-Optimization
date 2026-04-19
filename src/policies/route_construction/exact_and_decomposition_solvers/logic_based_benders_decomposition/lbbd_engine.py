import time
from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.lbbd import LBBDConfig
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

from .master import LBBDMasterProblem
from .subproblem import RoutingSubproblem


class LBBDEngine:
    """
    Coordination engine for Logic-Based Benders Decomposition.
    """

    def __init__(
        self,
        config: LBBDConfig,
        distance_matrix: np.ndarray,
        tree: ScenarioTree,
        capacity: float = 1.0,
    ):
        self.config = config
        self.distance_matrix = distance_matrix
        self.tree = tree
        self.capacity = capacity

        self.num_nodes = distance_matrix.shape[0]
        self.num_customers = self.num_nodes - 1

        self.master = LBBDMasterProblem(config, self.num_customers)
        self.subproblem = RoutingSubproblem(distance_matrix, config.subproblem_timeout)

        self.stats = {"iterations": 0, "cuts_added": 0, "converged": False, "solve_time": 0.0}

    def solve(self) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Runs the LBBD algorithm.
        Returns (full_plan, total_expected_profit, stats).
        """
        start_time = time.time()
        self.master.build(self.tree, self.config.stochastic_master)

        full_plan: List[List[List[int]]] = []
        total_profit = 0.0

        for iteration in range(1, self.config.max_iterations + 1):
            self.stats["iterations"] = iteration

            # 1. Solve Master Problem
            assignments, thetas = self.master.solve()
            if not assignments:
                break

            converged = True

            current_plan: List[List[List[int]]] = []

            # 2. Check each day for subproblem compliance
            for d in range(1, self.master.current_horizon + 1):
                nodes_in_d = assignments[d]

                # Solve Routing Subproblem for this assignment
                is_feasible, dist, route = self.subproblem.solve(nodes_in_d)

                # Store the route even if potentially infeasible (to avoid empty plan)
                current_plan.append([route] if route else [[0, 0]])

                if not is_feasible:
                    # Add Nogood cut
                    if self.config.use_nogood_cuts:
                        self.master.add_nogood_cut(d, nodes_in_d)
                        self.stats["cuts_added"] += 1
                        converged = False
                elif dist > thetas[d] + 1e-4 and self.config.use_optimality_cuts:
                    # Add Optimality cut
                    self.master.add_optimality_cut(d, nodes_in_d, dist)
                    self.stats["cuts_added"] += 1
                    converged = False

            # Update best plan
            full_plan = current_plan

            if converged:
                self.stats["converged"] = True
                break

            # Time limit check
            if time.time() - start_time > self.config.time_limit:
                break

        self.stats["solve_time"] = time.time() - start_time

        # Final objective value calculation on converged or best MP
        if self.master.model.SolCount > 0:  # type: ignore[union-attr]
            total_profit = self.master.model.ObjVal  # type: ignore[union-attr]

        return full_plan, total_profit, self.stats

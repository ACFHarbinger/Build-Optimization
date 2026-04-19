r"""
Integer L-Shaped (Benders Decomposition) Engine for Stochastic VRPP / SCWCVRP.

Implements the outer Benders decomposition loop that coordinates the Master
Problem (Gurobi MILP) and the analytical recourse subproblem evaluator.

Algorithm — Integer L-Shaped Method (Laporte & Louveaux, 1993):
    0. Generate S SAA scenarios {ω₁, …, ωₛ} of end-of-day bin fill levels.
    1. Build master problem with surrogate θ ≥ θ_LB.
    2. Repeat until convergence or iteration / time budget exhausted:
        a. Solve MP (with SEC lazy callbacks inside Gurobi) →
           integer routing (x̂, ŷ) and surrogate value θ̂.
        b. Evaluate Q̄(ŷ̂) analytically over SAA scenarios (no LP needed).
        c. If Q̄(ŷ̂) ≤ θ̂ + benders_gap: CONVERGED — break.
        d. Otherwise: generate optimality cut θ ≥ e + Σᵢ dᵢ yᵢ,
           add to MP, and re-solve.
    3. Return the best routing solution and its deterministic net profit.

Correctness Guarantees:
    - The subproblem is always feasible (affine in slack variables), so only
      optimality cuts are needed — no feasibility cuts.
    - Q̄(ŷ) is convex and piecewise-linear in ŷ; the gradient (d) is an exact
      supergradient, ensuring non-decreasing lower bounds across iterations.
    - Because Stage 1 variables are binary, the L-shaped cuts are *combinatorial*
      (valid at every integer point, not just at the LP relaxation).
    - The algorithm terminates finitely under a bound on the number of distinct
      binary ŷ solutions (exponential in |N| but bounded in practice by time/iter).

References:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel

from .master_problem import InventoryMasterProblem, MasterProblem
from .params import ILSBDParams
from .scenario import ScenarioGenerator
from .subproblem import RecourseEvaluator

try:
    import gurobipy  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class IntegerLShapedEngine:
    r"""Benders decomposition engine for stochastic VRPP / SCWCVRP.

    Orchestrates the full Integer L-Shaped method, including:
        - SAA scenario generation   (ScenarioGenerator)
        - Iterative master-problem solves  (MasterProblem / Gurobi)
        - Analytical recourse evaluation and cut generation (RecourseEvaluator)
        - Convergence monitoring and time-budget tracking

    Usage example::

        from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
        from logic.src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine import IntegerLShapedEngine
        from logic.src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params import ILSBDParams

        model = VRPPModel(n_nodes=..., cost_matrix=..., wastes=..., ...)
        engine = IntegerLShapedEngine(model, ILSBDParams())
        routes, y_hat, profit, stats = engine.solve(tree)

    Attributes:
        model: VRPPModel encoding the deterministic problem structure.
        params: ILSBDParams configuring all solver hyper-parameters.
        stats: Diagnostic dict populated during ``solve()``.
    """

    def __init__(self, model: VRPPModel, params: ILSBDParams) -> None:
        """Initialise the engine.

        Args:
            model: VRPPModel instance (network structure, costs, fill levels).
            params: ILSBDParams configuring Benders iterations, scenarios, penalties.
        """
        self.model = model
        self.params = params
        self.stats: Dict[str, Any] = {
            "benders_iterations": 0,
            "benders_cuts_added": 0,
            "converged": False,
            "total_time": 0.0,
            "final_Q_bar": None,
            "final_theta": None,
            "gurobi_available": GUROBI_AVAILABLE,
        }

        self._scenario_gen = ScenarioGenerator()
        self._evaluator = RecourseEvaluator()

    def solve(  # noqa: C901
        self,
        tree: Any,
        demand_matrix: Optional[np.ndarray] = None,
        bin_capacities: Optional[np.ndarray] = None,
        initial_inventory: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[int]], Dict[int, float], float, Dict[str, Any]]:
        r"""Run the (standard or Multi-Period) Integer L-Shaped Benders decomposition.

        Args:
            tree: A ScenarioTree object.
            demand_matrix: Optional mean demand increments per day per node.
            bin_capacities: Optional bin capacities per node.
            initial_inventory: Optional initial fill levels per node.

        Returns:
            Tuple of (routes, y_hat, profit, stats).
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi (gurobipy) is required for the Integer L-Shaped solver. ")

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 0: Extract Scenario Paths from Tree
        # ------------------------------------------------------------------
        from logic.src.pipeline.simulations.bins.prediction import ScenarioTreeNode

        def get_all_paths(node: ScenarioTreeNode, current_path: List[np.ndarray]) -> List[List[np.ndarray]]:
            new_path = current_path + [node.wastes]
            if not node.children:
                return [new_path]
            paths = []
            for child in node.children:
                paths.extend(get_all_paths(child, new_path))
            return paths

        scenarios = get_all_paths(tree.root, [])

        # ------------------------------------------------------------------
        # Step 1: Build master problem
        # ------------------------------------------------------------------
        is_multi_period = self.params.horizon > 1
        master: MasterProblem

        if is_multi_period:
            # Multi-period stochastic inventory routing
            N = len(self.model.customers)
            T = self.params.horizon

            # Build defaults if not provided
            if demand_matrix is None:
                demand_matrix = np.zeros((T, N), dtype=float)
                customers_list = list(self.model.customers)
                if hasattr(tree, "get_scenarios_at_day"):
                    for t in range(T):
                        scs = tree.get_scenarios_at_day(t) if t > 0 else []
                        if scs:
                            sc = scs[0]  # Mean scenario
                            for idx, node in enumerate(customers_list):
                                if hasattr(sc, "wastes") and node - 1 < len(sc.wastes):
                                    demand_matrix[t, idx] = float(sc.wastes[node - 1])

            bin_caps = bin_capacities if bin_capacities is not None else np.full(N, 100.0)
            init_inv = initial_inventory if initial_inventory is not None else np.zeros(N)

            master = InventoryMasterProblem(
                model=self.model,
                params=self.params,
                horizon=T,
                demand_matrix=demand_matrix,
                bin_capacities=bin_caps,
                initial_inventory=init_inv,
                stockout_penalty=self.params.stockout_penalty,
                big_m=self.params.big_m,
            )
            master.build()
            master.build_inventory()
        else:
            # Standard single-period stochastic VRPP
            master = MasterProblem(model=self.model, params=self.params)
            master.build()

        # Tracking state
        best_routes: List[List[int]] = []
        best_y_hat: Dict[int, float] = {}
        best_profit: float = -float("inf")
        self.stats["benders_iterations"] = 0
        self.stats["benders_cuts_added"] = 0
        self.stats["converged"] = False

        # ------------------------------------------------------------------
        # Step 2: Benders outer loop
        # ------------------------------------------------------------------
        for iteration in range(self.params.max_benders_iterations):
            self.stats["benders_iterations"] = iteration + 1

            # Respect wall-clock budget (keep 5 s reserve for extraction)
            elapsed = time.perf_counter() - t_start
            remaining = self.params.time_limit - elapsed
            if remaining < 5.0:
                break

            # Update Gurobi time limit to remaining wall-clock budget
            # we also respect master_time_limit if provided
            m_time = min(remaining, self.params.master_time_limit)
            master._gurobi_model.Params.TimeLimit = m_time  # type: ignore[union-attr]

            # ---- 2a. Solve master problem --------------------------------
            routes, y_hat, theta_hat, obj_MP = master.solve()

            # No feasible solution at all: terminate early
            if not routes and all(v == 0.0 for v in y_hat.values()):
                break

            # Track best deterministic solution (excludes θ from profit)
            det_profit = self._compute_deterministic_profit(routes, y_hat)
            if det_profit > best_profit:
                best_profit = det_profit
                best_routes = [list(r) for r in routes]
                best_y_hat = dict(y_hat)

            # ---- 2b. Evaluate recourse ----------------------------------
            Q_bar, e, d = self._evaluator.evaluate(
                y_hat=y_hat,
                scenarios=scenarios,
                overflow_penalty=self.params.overflow_penalty,
                undervisit_penalty=self.params.undervisit_penalty,
                collection_threshold=self.params.collection_threshold,
            )

            self.stats["final_Q_bar"] = Q_bar
            self.stats["final_theta"] = theta_hat

            if self.params.verbose:
                print(
                    f"[ILS] iter={iteration + 1:3d}  "
                    f"θ̂={theta_hat:10.4f}  Q̄={Q_bar:10.4f}  "
                    f"gap={Q_bar - theta_hat:+.4f}  "
                    f"cuts={self.stats['benders_cuts_added']}"
                )

            # ---- 2c. Convergence check ----------------------------------
            if Q_bar <= theta_hat + self.params.benders_gap:
                self.stats["converged"] = True
                break

            # ---- 2d. Add Benders optimality cut -------------------------
            master.add_optimality_cut(e=e, d=d)
            self.stats["benders_cuts_added"] += 1

        # ------------------------------------------------------------------
        # Step 3: Finalise statistics
        # ------------------------------------------------------------------
        self.stats["total_time"] = time.perf_counter() - t_start
        # Sync cut count from master's authoritative counter
        self.stats["benders_cuts_added"] = int(master.stats["benders_cuts"])

        if is_multi_period and isinstance(master, InventoryMasterProblem):
            self.stats["inventory_plan"] = master.get_inventory_plan()
            self.stats["collection_plan"] = master.get_collection_plan()
            self.stats["horizon"] = self.params.horizon

        return best_routes, best_y_hat, best_profit, self.stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_deterministic_profit(
        self,
        routes: List[List[int]],
        y_hat: Dict[int, float],
    ) -> float:
        r"""Compute the deterministic net profit Σᵢ rᵢ ŷᵢ − Σᵢⱼ cᵢⱼ x̂ᵢⱼ.

        The recourse surrogate θ is intentionally excluded so the returned
        value is compatible with the deterministic-policy contract used by
        BaseRoutingPolicy (``profit = revenue − travel_cost``).

        Args:
            routes: Customer-only route lists from the master problem.
            y_hat:  Binary visit decisions {node_idx: 0.0/1.0}.

        Returns:
            Net profit in the problem's monetary units.
        """
        revenue = sum(self.model.get_node_profit(i) * float(y_hat.get(i, 0.0)) for i in self.model.customers)

        travel_cost = 0.0
        for route in routes:
            if not route:
                continue
            path = [self.model.depot] + route + [self.model.depot]
            for k in range(len(path) - 1):
                travel_cost += self.model.get_edge_cost(path[k], path[k + 1])

        return revenue - travel_cost

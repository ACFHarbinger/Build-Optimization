r"""
Gurobi-based Master Problem for the Integer L-Shaped (Benders Decomposition) solver.

The Master Problem (MP) is a Mixed-Integer Linear Program containing:
    - Binary routing arc variables  x_{ij} ∈ {0,1}   (Stage 1 routing)
    - Binary node-visit variables   y_i    ∈ {0,1}   (Stage 1 selection)
    - Continuous surrogate variable θ  ≥ θ_LB         (expected recourse proxy)

Objective:
    min  Σᵢⱼ cᵢⱼ xᵢⱼ  −  Σᵢ rᵢ yᵢ  +  θ

Base feasibility constraints (built once at model construction):
    1. Degree:    Σⱼ x_{ij} = 2 yᵢ               ∀i ∈ Customers
    2. Depot:     Σⱼ x_{0j} ≤ 2K                  (fleet size K)
    3. Mandatory: yᵢ = 1                           ∀i ∈ mandatory
    4. Capacity:  Σᵢ wᵢ yᵢ ≤ K · Q               (aggregate knapsack)

Dynamic constraints (added iteratively between outer iterations):
    5. Benders optimality cuts: θ ≥ eₖ + Σᵢ dᵢₖ yᵢ  (one per Benders iteration)

Dynamic constraints (added inside Gurobi lazy/user callbacks):
    6. PC-Subtour Elimination Cuts (PC-SECs): routing feasibility
    7. Rounded Capacity Cuts (RCCs): per-route capacity enforcement

The model is built once via ``build()`` and re-solved iteratively.  Each call
to ``add_optimality_cut()`` inserts a new permanent Benders cut before the next
``solve()`` invocation.  Subtour elimination is handled by the same
``SeparationEngine`` used in the BranchAndCutSolver, keeping the SEC logic
consistent across all exact-solver policies.

References:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.

    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for
    the Symmetric Generalized Traveling Salesman Problem". Operations Research,
    45(2), 326-349.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel

from .params import ILSBDParams

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB = None  # type: ignore[assignment,misc,no-redef]


class MasterProblem:
    r"""Gurobi master problem for Integer L-Shaped Benders decomposition.

    Manages the MILP formulation with routing (x, y) and surrogate (θ)
    variables, lazy SEC / RCC callbacks for routing feasibility, and permanent
    Benders optimality cut injection between outer iterations.

    The model is built once (``build()``) and re-solved any number of times
    (``solve()``).  Each ``add_optimality_cut(e, d)`` call inserts a new regular
    Gurobi constraint before the next solve, preserving the accumulated Benders
    cuts and allowing Gurobi to warm-start from the previous incumbent.

    Attributes:
        model: VRPPModel encapsulating network structure, costs, and profits.
        params: ILSBDParams configuration.
        stats: Dict of accumulated solver statistics (cuts, nodes, time).
    """

    def __init__(self, model: VRPPModel, params: ILSBDParams) -> None:
        """Initialise the master problem container.

        Args:
            model: VRPPModel instance providing graph topology, costs, and profits.
            params: ILSBDParams controlling all solver parameters.

        Raises:
            ImportError: If gurobipy is not installed or not licensed.
        """
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi (gurobipy) is required for the Integer L-Shaped solver. "
                "Ensure gurobipy is installed and a valid licence is available."
            )

        self.model = model
        self.params = params

        # SEC / RCC separation engine (mirrors BranchAndCutSolver infrastructure)
        self.separator = SeparationEngine(
            model=self.model,
            enable_heuristic_rcc_separation=params.enable_heuristic_rcc_separation,
            enable_comb_cuts=params.enable_comb_cuts,
        )

        # Gurobi model handle and decision variable dicts
        self._gurobi_model: Optional[Any] = None
        self._x_vars: Dict[Tuple[int, int], Any] = {}
        self._y_vars: Dict[int, Any] = {}
        self._theta_var: Optional[Any] = None
        self._benders_cut_count: int = 0

        # Accumulated solver statistics (updated after each solve())
        self.stats: Dict[str, float] = {
            "sec_cuts": 0.0,
            "capacity_cuts": 0.0,
            "benders_cuts": 0.0,
            "nodes_explored": 0.0,
            "solve_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        r"""Construct the initial Gurobi MILP model.

        Creates decision variables (x, y, θ), the objective function, and the
        base deterministic constraints (degree, depot, mandatory, capacity).
        Subtour elimination cuts are deferred to the Gurobi lazy callback inside
        ``solve()``.  Call ``build()`` exactly once before the first ``solve()``.
        """
        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        self._gurobi_model = gp.Model("ILS_MasterProblem")
        m = self._gurobi_model

        m.Params.TimeLimit = self.params.time_limit
        m.Params.MIPGap = self.params.mip_gap
        m.Params.OutputFlag = 1 if self.params.verbose else 0

        # ---- Decision Variables ------------------------------------------

        # x_{ij} ∈ {0,1} (2 for depot edges to allow [0,i,0] single trips)
        for i, j in self.model.edges:
            ub = 2 if (i == self.model.depot or j == self.model.depot) else 1
            vtype = GRB.INTEGER if ub == 2 else GRB.BINARY
            self._x_vars[(i, j)] = m.addVar(lb=0, ub=ub, vtype=vtype, name=f"x_{i}_{j}")

        # y_i ∈ {0,1}: node i visited
        for i in self.model.customers:
            self._y_vars[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        # θ ∈ ℝ≥θ_LB: surrogate lower bound on expected recourse
        self._theta_var = m.addVar(
            lb=self.params.theta_lower_bound,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="theta",
        )

        # ---- Objective --------------------------------------------------
        # min  Σᵢⱼ cᵢⱼ xᵢⱼ  −  Σᵢ rᵢ yᵢ  +  θ
        travel_cost = gp.quicksum(self.model.get_edge_cost(i, j) * self._x_vars[(i, j)] for i, j in self.model.edges)
        revenue = gp.quicksum(self.model.get_node_profit(i) * self._y_vars[i] for i in self.model.customers)
        m.setObjective(travel_cost - revenue + self._theta_var, GRB.MINIMIZE)

        # ---- Constraints ------------------------------------------------

        # 1. Degree: Σⱼ x_{ij} = 2 yᵢ  ∀i ∈ Customers
        for i in self.model.customers:
            incident = [(u, v) for u, v in self.model.edges if i in (u, v)]
            m.addConstr(
                gp.quicksum(self._x_vars[e] for e in incident) == 2 * self._y_vars[i],
                name=f"degree_{i}",
            )

        # 2. Depot degree: Σⱼ x_{0j} ≤ 2K
        depot_edges = [(u, v) for u, v in self.model.edges if self.model.depot in (u, v)]
        m.addConstr(
            gp.quicksum(self._x_vars[e] for e in depot_edges) <= 2 * self.model.num_vehicles,
            name="depot_degree",
        )

        # 3. Mandatory nodes: yᵢ = 1  ∀i ∈ mandatory
        for i in self.model.mandatory_nodes:
            if i in self._y_vars:
                m.addConstr(self._y_vars[i] == 1, name=f"mandatory_{i}")

        # 4. Global capacity: Σᵢ wᵢ yᵢ ≤ K · Q
        m.addConstr(
            gp.quicksum(self.model.get_node_demand(i) * self._y_vars[i] for i in self.model.customers)
            <= self.model.num_vehicles * self.model.capacity,
            name="global_capacity",
        )

        m.update()

    def add_optimality_cut(self, e: float, d: Dict[int, float]) -> None:
        r"""Add a Benders L-shaped optimality cut: θ ≥ e + Σᵢ dᵢ yᵢ.

        The cut is added as a permanent regular Gurobi constraint (not lazily
        inside a callback), so it is enforced at every subsequent B&B node.

        Args:
            e: Constant term of the Benders cut.
            d: Per-node coefficients {node_idx: dᵢ} for the linear yᵢ terms.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
        """
        if self._gurobi_model is None or self._theta_var is None:
            raise RuntimeError("MasterProblem.build() must be called before add_optimality_cut().")

        assert GUROBI_AVAILABLE and gp is not None

        # θ ≥ e + Σᵢ dᵢ yᵢ
        cut_lhs = gp.quicksum(d[i] * self._y_vars[i] for i in d if i in self._y_vars)
        self._benders_cut_count += 1
        self._gurobi_model.addConstr(
            self._theta_var >= e + cut_lhs,
            name=f"benders_cut_{self._benders_cut_count}",
        )
        self._gurobi_model.update()
        self.stats["benders_cuts"] = float(self._benders_cut_count)

    def solve(self) -> Tuple[List[List[int]], Dict[int, float], float, float]:
        r"""Solve the master problem with the current set of Benders cuts.

        Runs Gurobi's branch-and-cut with lazy SEC / RCC callbacks for routing
        feasibility.  Returns the best integer solution found within the current
        ``params.time_limit`` (which the calling engine updates between iterations).

        Returns:
            Tuple of (routes, y_hat, theta_hat, obj_MP):
                routes: List of customer-only route lists (depot index excluded).
                         Empty list if no feasible solution was found.
                y_hat:  Dict {node_idx: 0.0 or 1.0} visit decisions.
                theta_hat: Value of the surrogate variable θ in the optimal solution.
                obj_MP:    Master problem objective value (travel_cost − revenue + θ).
        """
        assert self._gurobi_model is not None
        assert GUROBI_AVAILABLE and GRB is not None

        m = self._gurobi_model
        # CRITICAL: LazyConstraints=1 is mandatory; without it Gurobi ignores cbLazy
        m.Params.LazyConstraints = 1
        m.Params.PreCrush = 1  # Allow user cuts to strengthen presolve
        m.optimize(self._lazy_constraint_callback)

        self.stats["nodes_explored"] = float(m.NodeCount)
        self.stats["solve_time"] = float(m.Runtime)

        if m.SolCount == 0:
            return (
                [],
                {i: 0.0 for i in self.model.customers},
                self.params.theta_lower_bound,
                float("inf"),
            )

        routes = self._extract_routes()
        y_hat = {i: float(round(self._y_vars[i].X)) for i in self.model.customers}
        theta_hat = float(self._theta_var.X)  # type: ignore[union-attr]
        obj_MP = float(m.ObjVal)

        return routes, y_hat, theta_hat, obj_MP

    # ------------------------------------------------------------------
    # Gurobi callbacks
    # ------------------------------------------------------------------

    def _lazy_constraint_callback(self, model: Any, where: int) -> None:
        """Gurobi lazy/user cut callback for SEC and RCC separation.

        Dispatches to integer-cut separation at MIPSOL events and to fractional
        LP-cut separation at MIPNODE events, mirroring the BranchAndCutSolver
        callback structure.

        Args:
            model: Gurobi model reference passed by the optimize() framework.
            where: Gurobi callback location constant (GRB.Callback.*).
        """
        assert GRB is not None

        if where == GRB.Callback.MIPSOL:
            self._add_integer_cuts(model)
        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            self._add_fractional_cuts(model)

    def _add_integer_cuts(self, model: Any) -> None:
        """Separate and add violated SEC / RCC cuts at an integer solution (MIPSOL).

        Args:
            model: Gurobi model handle from the callback.
        """
        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        x_var_list = [self._x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self._y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetSolution(x_var_list))
        y_vals = np.array(model.cbGetSolution(y_var_list))

        cuts = self.separator.separate_integer(
            x_vals,
            y_vals=y_vals,
            max_cuts=self.params.max_cuts_per_round,
            iteration=int(self.stats["sec_cuts"]),
            sec_only=False,
        )

        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    if cut.facet_form == "2.1":
                        model.cbLazy(gp.quicksum(edge_vars) >= 2.0)
                    elif cut.facet_form == "2.2":
                        yi = self._y_vars[cut.node_i]
                        model.cbLazy(gp.quicksum(edge_vars) >= 2 * yi)
                    else:  # Form 2.3
                        yi = self._y_vars[cut.node_i]
                        yj = self._y_vars.get(cut.node_j, 1.0)
                        model.cbLazy(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))
                self.stats["sec_cuts"] += 1

            elif isinstance(cut, CapacityCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)
                self.stats["capacity_cuts"] += 1

    def _add_fractional_cuts(self, model: Any) -> None:
        """Separate and add user cuts at a fractional LP node (MIPNODE).

        Args:
            model: Gurobi model handle from the callback.
        """
        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        x_var_list = [self._x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self._y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetNodeRel(x_var_list))
        y_vals = np.array(model.cbGetNodeRel(y_var_list))

        node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
        cuts = self.separator.separate_fractional(
            x_vals,
            y_vals=y_vals,
            max_cuts=self.params.max_cuts_per_round // 2,
            iteration=int(self.stats["sec_cuts"]),
            node_count=node_count,
        )

        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    if cut.facet_form == "2.1":
                        model.cbCut(gp.quicksum(edge_vars) >= 2.0)
                    elif cut.facet_form == "2.2":
                        yi = self._y_vars[cut.node_i]
                        model.cbCut(gp.quicksum(edge_vars) >= 2 * yi)
                    else:  # Form 2.3
                        yi = self._y_vars[cut.node_i]
                        yj = self._y_vars.get(cut.node_j, 1.0)
                        model.cbCut(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

            elif isinstance(cut, CapacityCut):
                cut_edges = self.model.delta(cut.node_set)
                edge_vars = [self._x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self._x_vars]  # type: ignore[bad-index,index]
                if edge_vars:
                    model.cbCut(gp.quicksum(edge_vars) >= cut.rhs)

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _extract_routes(self) -> List[List[int]]:
        """Extract customer-only route lists from the current Gurobi solution.

        Traces all vehicle paths starting from the depot by following the
        active arc edges.  Matches the multi-vehicle extraction logic in
        BranchAndCutSolver._extract_solution().

        Returns:
            List of routes, each a list of customer indices (depot excluded).
        """
        assert self._gurobi_model is not None

        if self._gurobi_model.SolCount == 0:
            return []

        # Build a multiset of active (sorted) edges
        active_edges: List[List[int]] = []
        for (i, j), var in self._x_vars.items():
            val = round(var.X)
            for _ in range(int(val)):
                active_edges.append([min(i, j), max(i, j)])

        if not active_edges:
            return []

        routes: List[List[int]] = []
        remaining = list(active_edges)

        while True:
            # Find a depot-incident edge to start a new route
            start_edge = next((e for e in remaining if self.model.depot in e), None)
            if start_edge is None:
                break

            route = [self.model.depot]
            current = start_edge[1] if start_edge[0] == self.model.depot else start_edge[0]
            remaining.remove(start_edge)
            route.append(current)

            while current != self.model.depot:
                found = False
                for e in remaining:
                    if current in e:
                        nxt = e[1] if e[0] == current else e[0]
                        remaining.remove(e)
                        route.append(nxt)
                        current = nxt
                        found = True
                        break
                if not found:
                    break

            customers_only = [n for n in route if n != self.model.depot]
            if customers_only:
                routes.append(customers_only)

        return routes


class InventoryMasterProblem(MasterProblem):
    r"""Gurobi master problem with inventory balance linking constraints.

    Extends the single-period ``MasterProblem`` to handle a T-day horizon by
    adding :math:`I_{it}`, :math:`q_{it}`, and overflow slack variables with
    the corresponding inventory balance and coupling constraints.

    The additional variables are indexed ``(node_id, day)``, where
    ``day ∈ {0, …, T-1}``.

    Attributes:
        T: Planning horizon (number of days).
        demand_matrix: Array of shape ``(T, N)`` with expected demand increments
            per day per node (mean scenario).
        bin_capacities: Array of shape ``(N,)`` with capacity (100.0 for %).
        initial_inventory: Array of shape ``(N,)`` with initial fill levels.
        stockout_penalty: Penalty per unit of overflow slack.
        big_m: Big-M constant for collection coupling constraints.
    """

    def __init__(
        self,
        model: VRPPModel,
        params: ILSBDParams,
        horizon: int = 7,
        demand_matrix: Optional[np.ndarray] = None,
        bin_capacities: Optional[np.ndarray] = None,
        initial_inventory: Optional[np.ndarray] = None,
        stockout_penalty: float = 500.0,
        big_m: float = 1e4,
    ) -> None:
        r"""Initialise the inventory-extended master problem.

        Args:
            model: VRPPModel providing graph topology and costs.
            params: ILSBDParams configuration.
            horizon: Planning horizon T.
            demand_matrix: Expected demand increments, shape ``(T, N)``.
                           If None, defaults to zeros (no demand evolution).
            bin_capacities: Bin capacities per node (% units), shape ``(N,)``.
                            If None, defaults to 100.0 for all nodes.
            initial_inventory: Initial fill levels, shape ``(N,)``.
                               If None, defaults to zeros.
            stockout_penalty: Penalty per unit overflow per day.
            big_m: Big-M constant for :math:`q_{it} \leq M \cdot y_{it}`.
        """
        super().__init__(model=model, params=params)

        self.T = horizon
        N = len(model.customers)

        self.demand_matrix: np.ndarray = (
            demand_matrix if demand_matrix is not None else np.zeros((horizon, N), dtype=float)
        )
        self.bin_capacities: np.ndarray = (
            bin_capacities if bin_capacities is not None else np.full(N, 100.0, dtype=float)
        )
        self.initial_inventory: np.ndarray = (
            initial_inventory if initial_inventory is not None else np.zeros(N, dtype=float)
        )
        self.stockout_penalty = stockout_penalty
        self.big_m = big_m

        # Inventory decision variables: keyed (node_id, day)
        self._I_vars: Dict[Tuple[int, int], Any] = {}
        self._q_vars: Dict[Tuple[int, int], Any] = {}
        self._s_vars: Dict[Tuple[int, int], Any] = {}  # overflow slack

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_inventory(self) -> None:
        """Extend the base master model with inventory balance constraints.

        Must be called **after** ``build()`` so that the base variables
        (:math:`x`, :math:`y`, :math:`\\theta`) already exist.

        Raises:
            RuntimeError: If ``build()`` has not been called first.
            ImportError: If gurobipy is unavailable.
        """
        if self._gurobi_model is None:
            raise RuntimeError("InventoryMasterProblem.build() must be called before build_inventory().")

        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        m = self._gurobi_model
        customers = list(self.model.customers)

        # ---- Inventory decision variables --------------------------------
        for idx, i in enumerate(customers):
            cap_i = float(self.bin_capacities[idx]) if idx < len(self.bin_capacities) else 100.0
            for t in range(self.T):
                # I_{it}: inventory at end of day t
                self._I_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"I_{i}_{t}",
                )
                # q_{it}: collected quantity on day t
                self._q_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=cap_i,
                    vtype=GRB.CONTINUOUS,
                    name=f"q_{i}_{t}",
                )
                # s_{it}: overflow slack (>= 0)
                self._s_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"s_{i}_{t}",
                )

        m.update()

        # ---- Inventory balance & coupling constraints --------------------
        for idx, i in enumerate(customers):
            # yi_var = self._y_vars.get(i) # Use the Day-0 visit decisions (Stage 1)
            # Actually, the base class has self._y_vars
            yi_var = self._y_vars.get(i)

            for t in range(self.T):
                cap_i = float(self.bin_capacities[idx]) if idx < len(self.bin_capacities) else 100.0
                demand_it = float(self.demand_matrix[t, idx]) if t < self.demand_matrix.shape[0] else 0.0

                I_prev = self.initial_inventory[idx] if t == 0 else self._I_vars[(i, t - 1)]

                # 1. Inventory balance: I_{it} = I_{i,t-1} + d_{it} - q_{it}
                m.addConstr(
                    self._I_vars[(i, t)] == I_prev + demand_it - self._q_vars[(i, t)],
                    name=f"inv_balance_{i}_{t}",
                )

                # 2. Coupling: q_{it} ≤ M · y_{it}
                # We use the Day-0 visit decisions (y_i) as the Stage-1 variable.
                if yi_var is not None:
                    m.addConstr(
                        self._q_vars[(i, t)] <= self.big_m * yi_var,
                        name=f"coupling_{i}_{t}",
                    )

                # 3. Capacity bound + overflow linearisation:
                #    I_{it} - s_{it} ≤ B_i  →  s_{it} ≥ I_{it} - B_i
                m.addConstr(
                    self._I_vars[(i, t)] - self._s_vars[(i, t)] <= cap_i,
                    name=f"overflow_{i}_{t}",
                )

        # ---- Extend objective with overflow penalty ----------------------
        overflow_cost = gp.quicksum(
            self.stockout_penalty * self._s_vars[(i, t)] for i in customers for t in range(self.T)
        )
        current_obj = m.getObjective()
        m.setObjective(current_obj + overflow_cost, GRB.MINIMIZE)
        m.update()

    def get_inventory_plan(self) -> Dict[Tuple[int, int], float]:
        """Extract the optimal inventory plan after solve().

        Returns:
            Dict mapping ``(node_id, day)`` to inventory level :math:`I_{it}`.
        """
        if self._gurobi_model is None or self._gurobi_model.SolCount == 0:
            return {}
        return {(i, t): float(var.X) for (i, t), var in self._I_vars.items()}

    def get_collection_plan(self) -> Dict[Tuple[int, int], float]:
        """Extract the optimal collection plan after solve().

        Returns:
            Dict mapping ``(node_id, day)`` to collected quantity :math:`q_{it}`.
        """
        if self._gurobi_model is None or self._gurobi_model.SolCount == 0:
            return {}
        return {(i, t): float(var.X) for (i, t), var in self._q_vars.items()}

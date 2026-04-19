r"""
Branch-and-Cut Solver for VRPP using Gurobi.

This solver adapts the mathematical infrastructure of Fischetti et al. (1997)
for the Symmetric Generalized TSP to the Single-Vehicle VRPP. Nodes are treated
as Singleton Clusters, and SECs are implemented as Generalized Subtour Elimination
Constraints (GSECs) which reduce to Prize-Collecting SECs (PC-SECs).

References:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.

    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut algorithm
    for the capacitated vehicle routing problem". Mathematical Programming, 100(2), 423-445.
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics import (
    construct_initial_solution,
    construct_nn_solution,
    farthest_insertion,
)

from .params import BCParams

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[assignment,misc,no-redef]


try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class BranchAndCutSolver:
    """
    Branch-and-Cut solver for VRPP.

    Follows the cutting plane algorithm described in Section 5.1 of Fischetti et al. (1997):
    1. Solve LP relaxation
    2. Separate violated inequalities
    3. Add cuts and resolve
    4. Branch if fractional
    5. Use primal heuristics for upper bounds
    """

    def __init__(
        self,
        model: VRPPModel,
        params: Optional[BCParams] = None,
        scenarios: Optional[List[Dict[int, float]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize Branch-and-Cut solver.

        Args:
            model: VRPPModel instance defining the problem.
            params: Standardized BC configuration.
            scenarios: Optional list of mappings from node index to stochastic demand for SAA scenarios.
            **kwargs: Legacy configuration parameters (for backward compatibility).
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for Branch-and-Cut solver")

        if params is None:
            # Create a defaults-aware params object from explicit arguments or framework defaults
            params = BCParams(
                time_limit=kwargs.get("time_limit", 300.0),
                mip_gap=kwargs.get("mip_gap", 0.01),
                max_cuts_per_round=kwargs.get("max_cuts_per_round", 50),
                use_heuristics=kwargs.get("use_heuristics", True),
                verbose=kwargs.get("verbose", False),
                profit_aware_operators=kwargs.get("profit_aware_operators", False),
                vrpp=kwargs.get("vrpp", False),
                enable_fractional_capacity_cuts=kwargs.get("enable_fractional_capacity_cuts", True),
            )

        self.model = model
        self.params = params

        # SAA Scenario Handling Scaffold
        self.scenarios = scenarios
        if self.params.use_saa:
            if self.params.verbose:
                print("============================================================")
                print("⚠ WARNING: Solving massive deterministic equivalent MILP   ")
                print("  for SAA. LP relaxations for compact stochastic models ")
                print("  are notoriously weak and solver may choke on symmetry.")
                print("============================================================")
            if not self.scenarios:
                if self.params.verbose:
                    print(f"Generating {self.params.num_scenarios} mock scenarios for SAA scaffold.")
                self.scenarios = []
                for _ in range(self.params.num_scenarios):
                    scen = {}
                    for idx in self.model.customers:
                        base_demand = self.model.get_node_demand(idx)
                        scen[idx] = max(0.0, np.random.normal(base_demand, max(0.01, 0.2 * base_demand)))
                    self.scenarios.append(scen)

        # Separation engine with adaptive capacity cut toggling
        # Auto-disable for large instances (n > 75) to prevent O(V⁴) bottleneck
        if model.n_nodes > 75 and self.params.enable_fractional_capacity_cuts:
            if self.params.verbose:
                print(f"⚠ Large instance detected (n={model.n_nodes}). Disabling exact fractional capacity cuts.")
            self.params.enable_fractional_capacity_cuts = False

        self.separator = SeparationEngine(
            model=self.model,
            enable_heuristic_rcc_separation=self.params.enable_heuristic_rcc_separation,
            enable_comb_cuts=getattr(self.params, "enable_comb_cuts", False),
        )

        # Statistics
        self.stats = {
            "total_cuts": 0,
            "sec_cuts": 0,
            "capacity_cuts": 0,
            "nodes_explored": 0,
            "lp_iterations": 0,  # counts integer solutions (MIPSOL events)
            "fractional_iterations": 0.0,  # counts fractional LP nodes (MIPNODE events)
        }

        # Cut pool: stores compact representations of previously found violated cuts.
        # Paper (Section 5.1): "All violated constraints found (except fan inequalities)
        # are permanently stored in compact form in a global data structure called the pool."
        # At each new B&B node, pool cuts are re-evaluated first before running any
        # separation algorithm, avoiding redundant rediscovery of ancestor cuts.
        # NOTE: Fan inequality separation (Equation 2.12) is NOT implemented.
        # NOTE: Pool re-evaluation is active via _evaluate_pool_cuts(), called at
        #       the start of both _add_integer_cuts() and _add_fractional_cuts().
        self._cut_pool: List[Any] = []
        self._cut_signatures: Set[Tuple[Any, ...]] = set()

        # Gurobi model
        self.gurobi_model: Optional[gp.Model] = None
        self.x_vars: Dict[Tuple[int, int], gp.Var] = {}
        self.y_vars: Dict[int, gp.Var] = {}

    def solve(self) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        """
        Solve the VRPP instance.

        Returns:
            Tuple of (tour, profit, statistics).
        """
        if self.params.verbose:
            print("=" * 60)
            print("Branch-and-Cut Solver for VRPP")
            print("=" * 60)
            print(f"Nodes: {self.model.n_nodes}")
            print(f"Capacity: {self.model.capacity}")
            print(f"Mandatory nodes: {len(self.model.mandatory_nodes)}")
            print("=" * 60)

        # Step 1: Build initial Gurobi model
        self._build_initial_model()

        # Step 2: Root node strengthening via Lagrangian relaxation.
        # Return value intentionally discarded — used only for VarHintVal hints.
        self._pre_optimize_lagrangian()

        # Step 3: REMOVED (Lagrangian GSECs are now separated in the first LP callback)

        # Step 4: Get initial primal solution (heuristic)
        if self.params.use_heuristics:
            best_tour = []
            best_profit = -float("inf")

            # 1. Greedy Profit Initialization (Best for VRPP Knapsack constraints)
            tour_greedy, profit_greedy = construct_initial_solution(self.model)
            if profit_greedy > best_profit:
                best_profit, best_tour = profit_greedy, tour_greedy

            # 2. Nearest Neighbor Initialization (Good for tight spatial clustering)
            tour_nn, profit_nn = construct_nn_solution(self.model)
            if profit_nn > best_profit:
                best_profit, best_tour = profit_nn, tour_nn

            # 3. Farthest Insertion (Good for exploring the convex hull, as per Fischetti 1997)
            tour_farthest, profit_farthest = farthest_insertion(
                self.model,
                profit_aware_operators=self.params.profit_aware_operators,
                expand_pool=self.params.vrpp,
            )
            if profit_farthest > best_profit:
                best_profit, best_tour = profit_farthest, tour_farthest

            if best_tour:
                if self.params.verbose:
                    print(f"Warm start heuristic selected with profit: {best_profit:.2f}")
                self._set_start_solution(best_tour)

        # Step 3: Enable lazy constraint callback for cutting planes
        # CRITICAL: LazyConstraints=1 is MANDATORY for correctness.
        # Without MTZ variables, the base model CANNOT enforce subtours or capacity on its own.
        # The SeparationEngine dynamically detects and adds violated SEC/RCC cuts via callbacks.
        # If callbacks fail or are disabled, the solver will return invalid routes.
        assert self.gurobi_model is not None
        self.gurobi_model.Params.LazyConstraints = 1
        self.gurobi_model.Params.PreCrush = 1  # type: ignore[union-attr]
        self.gurobi_model.optimize(self._lazy_constraint_callback)  # type: ignore[union-attr]

        # Step 4: Extract solution
        routes, profit = self._extract_solution()

        self.stats["obj_value"] = profit
        assert self.gurobi_model is not None
        self.stats["solve_time"] = self.gurobi_model.Runtime
        self.stats["mip_gap"] = self.gurobi_model.MIPGap if self.gurobi_model.SolCount > 0 else 1.0
        self.stats["nodes_explored"] = int(self.gurobi_model.NodeCount)

        return routes, profit, self.stats  # type: ignore[return-value]

    def _build_initial_model(self):
        """Build the initial Gurobi model with basic constraints."""
        self.gurobi_model = gp.Model("VRPP_BranchAndCut")
        self.gurobi_model.Params.TimeLimit = self.params.time_limit
        self.gurobi_model.Params.MIPGap = self.params.mip_gap
        self.gurobi_model.Params.OutputFlag = 1 if self.params.verbose else 0

        # Decision variables
        # x[i,j]: Edge (i,j) is in the tour
        for i, j in self.model.edges:
            # Special case for depot edges: allow value 2 for [0, i, 0] trips
            if i == self.model.depot or j == self.model.depot:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(lb=0, ub=2, vtype=GRB.INTEGER, name=f"x_{i}_{j}")
            else:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # y[i]: Node i is visited
        for i in self.model.customers:
            self.y_vars[i] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        # Objective: Maximize profit = waste collected - travel cost
        # Gurobi minimizes by default, so negate
        waste_collected = gp.quicksum(self.model.get_node_profit(i) * self.y_vars[i] for i in self.model.customers)

        travel_cost = gp.quicksum(self.model.get_edge_cost(i, j) * self.x_vars[(i, j)] for i, j in self.model.edges)

        if self.params.use_saa and self.scenarios:
            self.q_vars = {}
            num_s = len(self.scenarios)
            recourse_cost_expr = 0.0

            # SIRP Scaffold: Recourse cost is penalized excess capacity per scenario.
            # In a full recourse formulation, this might measure failure returns to depot.
            for s_idx, scenario_demands in enumerate(self.scenarios):
                q_s = self.gurobi_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_recourse_{s_idx}")
                self.q_vars[s_idx] = q_s

                # Link recourse to routing baseline decisions
                scenario_load = gp.quicksum(scenario_demands[i] * self.y_vars[i] for i in self.model.customers)
                self.gurobi_model.addConstr(
                    q_s >= scenario_load - (self.model.num_vehicles * self.model.capacity), name=f"recourse_def_{s_idx}"
                )

                # High penalty for capacity violation in scenario (e.g. 10 units of cost)
                recourse_cost_expr += 10.0 * self.model.C * q_s

            expected_recourse = recourse_cost_expr / num_s
            self.gurobi_model.setObjective(travel_cost - waste_collected + expected_recourse, GRB.MINIMIZE)
        else:
            self.gurobi_model.setObjective(travel_cost - waste_collected, GRB.MINIMIZE)

        # Constraints
        # 1. Degree constraints: sum of edges incident to i equals 2*y[i]
        for i in self.model.customers:
            incident_edges = [(u, v) for u, v in self.model.edges if i in (u, v)]
            self.gurobi_model.addConstr(
                gp.quicksum(self.x_vars[e] for e in incident_edges) == 2 * self.y_vars[i],
                name=f"degree_{i}",
            )

        # Depot degree constraint (allows up to K vehicles)
        depot_edges = [(u, v) for u, v in self.model.edges if self.model.depot in (u, v)]
        self.gurobi_model.addConstr(
            gp.quicksum(self.x_vars[e] for e in depot_edges) <= 2 * self.model.num_vehicles,
            name="depot_degree",
        )

        # 2. Mandatory node constraints
        for i in self.model.mandatory_nodes:
            if i in self.y_vars:
                self.gurobi_model.addConstr(self.y_vars[i] == 1, name=f"mandatory_{i}")

        # 3. Global capacity constraint (Knapsack)
        # sum of demands of visited nodes <= K * Q (aggregate capacity)
        self.gurobi_model.addConstr(
            gp.quicksum(self.model.get_node_demand(i) * self.y_vars[i] for i in self.model.customers)
            <= self.model.num_vehicles * self.model.capacity,
            name="global_capacity",
        )

        self.gurobi_model.update()

    def _lazy_constraint_callback(self, model, where):
        """
        Gurobi callback for Optimized Branch-and-Cut.

        Optimized Callback Structure:
            1. MIPSOL (Integer Solutions):
               - Only separates and adds Subtour Elimination Cuts (SECs).
               - These are mandatory for integer feasibility to prevent disconnected cycles.
               - Added via cbLazy().

            2. MIPNODE (Fractional LP Nodes):
               - Separates and adds both Capacity Cuts and SECs.
               - These act as User Cuts strengthening the LP relaxation bound.
               - Capacity Cuts are mathematically redundant for integer feasibility
                 due to the global knapsack constraint, but essential for a tight LP.
               - Added via cbCut().
        """
        if where == GRB.Callback.MIPSOL:
            # Integer solution: only SECs are mandatory for feasibility
            self._add_integer_cuts(model)

        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Fractional LP node: add both SECs and Capacity Cuts as User Cuts
            self._add_fractional_cuts(model)

    def _evaluate_pool_cuts(self, x_vals: np.ndarray, y_vals: np.ndarray) -> List[Any]:
        """
        Re-evaluate stored cuts against the current LP/MIP solution.
        Returns cuts whose violation exceeds the threshold.
        Paper Section 5.1: pool cuts are checked before any separation call.
        """
        active = []
        for cut in self._cut_pool:
            if isinstance(cut, PCSubtourEliminationCut):
                # node_i and node_j must be customers (> 0) for non-2.1 forms.
                # The depot cannot appear in S for any valid SEC.
                assert cut.node_i != 0 or cut.facet_form == "2.1", (
                    f"Malformed pool cut: node_i=0 for facet_form={cut.facet_form}"
                )
                cut_val = self.separator._get_cut_value(cut.node_set, x_vals)
                if cut.facet_form == "2.1":
                    violation = 2.0 - cut_val
                elif cut.facet_form == "2.2":
                    yi = y_vals[cut.node_i - 1] if cut.node_i > 0 else 1.0
                    violation = 2.0 * yi - cut_val
                else:
                    yi = y_vals[cut.node_i - 1] if cut.node_i > 0 else 1.0
                    yj = y_vals[cut.node_j - 1] if cut.node_j > 0 else 1.0
                    violation = 2.0 * (yi + yj - 1.0) - cut_val
                if violation > 0.01:
                    active.append(cut)
            elif isinstance(cut, CapacityCut):
                cut_val = self.separator._get_cut_value(cut.node_set, x_vals)
                violation = cut.rhs - cut_val
                if violation > 0.01:
                    active.append(cut)  # type: ignore[arg-type]
        return active

    def _handle_custom_branching(self, model):
        """
        Custom branching on cuts (Section 5.4, Fischetti et al. 1997).

        NOT IMPLEMENTED: Gurobi does not expose a programmatic branching
        interface from within Python callbacks. The paper's "branching on cuts"
        strategy (choosing S such that sum_{e in delta(S)} x_e is fractional-odd
        and imposing sum x_e <= 2k or sum x_e >= 2k+2) cannot be enforced via
        cbBranch() — no such Gurobi API method exists.

        Alternative approaches for influencing branching in Gurobi:
          1. Set BranchPriority on x_vars to prefer edges near 0.5.
          2. Use GRB.Param.BranchDir to prefer branching directions.
          3. Implement a custom B&B loop outside Gurobi (not done here).

        Gurobi's default variable branching (most fractional variable) is used
        instead, which corresponds to the paper's fallback strategy.
        """
        pass

    def _add_integer_cuts(self, model):
        """
        Add violated cuts at integer solutions (MIPSOL callback).

        This method ensures correctness for the Natural Edge Formulation:
        1. Extracts the current integer solution (x_vals, y_vals) via cbGetSolution()
        2. Invokes the SeparationEngine to detect violated inequalities (fast heuristics)
        3. Adds violated cuts as lazy constraints via model.cbLazy()

        Correctness Contract (Fischetti et al. 1997):
            - If the solution contains a disconnected subtour, the SeparationEngine
              MUST detect it and return a SubtourEliminationCut.
            - If the solution violates vehicle capacity constraints for any node set,
              the SeparationEngine MUST detect it and return a CapacityCut.
            - If no violated cuts are found, the solution is guaranteed to be valid.

        Performance:
            - Uses fast connected-components check for SECs (NetworkX-based)
            - Heuristic capacity separation (demand-based clustering)
            - O(n²) complexity, suitable for integer callback
        """
        # Get current integer solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetSolution() reduces Python-C API overhead
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetSolution(x_var_list))
        y_vals_array = np.array(model.cbGetSolution(y_var_list))

        # Re-evaluate pool cuts before fresh separation
        pool_cuts = self._evaluate_pool_cuts(x_vals, y_vals_array)
        for cut in pool_cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_lazy(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_lazy(model, cut)
                self.stats["capacity_cuts"] += 1
            self.stats["total_cuts"] += 1

        # Separate both SECs and Capacity Cuts at integer solutions.
        # Capacity cuts are required for correctness: a global knapsack constraint
        # does NOT prevent individual routes from violating per-route capacity.
        # sec_only=False ensures capacity-violating integer solutions are rejected.
        iteration = self.stats["lp_iterations"]
        cuts = self.separator.separate_integer(
            x_vals, y_vals=y_vals_array, max_cuts=self.params.max_cuts_per_round, iteration=iteration, sec_only=False
        )

        # Add cuts as lazy constraints
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_lazy(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_lazy(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        # Store cuts for pool re-evaluation at child nodes (active via _evaluate_pool_cuts).
        for cut in cuts:
            # Signature: (node_set, class_name, facet_form, node_i, node_j)
            node_set = getattr(cut, "node_set", set())
            sig = (
                frozenset(node_set) if node_set else frozenset(),
                cut.__class__.__name__,
                getattr(cut, "facet_form", None),
                getattr(cut, "node_i", None),
                getattr(cut, "node_j", None),
            )
            if sig not in self._cut_signatures:
                self._cut_pool.append(cut)
                self._cut_signatures.add(sig)

        # SAA SIRP separation scaffolds for integer feasibility
        if self.params.use_saa and self.scenarios:
            self._separate_stochastic_capacity_cuts(model, x_vals, y_vals_array, is_integer=True)
            self._separate_multi_star_inequalities(model, x_vals, y_vals_array, is_integer=True)
            self._separate_lot_sizing_inequalities(model, x_vals, y_vals_array, is_integer=True)

        self.stats["lp_iterations"] += 1

    def _add_fractional_cuts(self, model):
        """
        Add violated cuts at fractional LP relaxation nodes (MIPNODE callback).

        DIVERGENCE FROM PAPER (Section 5.1, procedure SEPARATION):
        The paper separates in this order:
          1. Pool re-evaluation (cuts found at ancestor nodes)
          2. Fan inequalities (Equation 2.12) — NOT IMPLEMENTED HERE
          3. GSEC_H2 heuristic
          4. GSEC_H1 heuristic
          5. Per-threshold ε ∈ {0.1, 0.01, 0.001} combined pool check
          6. GSEC_SEP exact separation (forced every 10th node)
          7. Generalised comb heuristics — NOT IMPLEMENTED HERE
        This implementation calls separate_fractional() which covers steps 3, 4,
        and 6 (throttled). Steps 2, 5, and 7 are not implemented.
        Paper Rule (Section 3.1): Exact separation (GSEC_SEP) is forced at every
        10th decision-tree node regardless of depth.
        """
        # Get current node count (depth indicator)
        node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

        # Get fractional solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetNodeRel() is O(n) vs O(n²) for loop-based extraction
        # This reduces Python-C API overhead from ~1000 calls to 2 calls for n=50 instances
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetNodeRel(x_var_list))
        y_vals_array = np.array(model.cbGetNodeRel(y_var_list))

        # Re-evaluate pool cuts before fresh separation
        pool_cuts = self._evaluate_pool_cuts(x_vals, y_vals_array)
        for cut in pool_cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_user(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_user(model, cut)
                self.stats["capacity_cuts"] += 1
            self.stats["total_cuts"] += 1

        # Separate violated inequalities (fractional mode: exact max-flow separation)
        self.stats["fractional_iterations"] += 1
        iteration = int(self.stats["fractional_iterations"])
        max_cuts = self.params.max_cuts_per_round // 2  # Limit cuts at fractional nodes

        cuts = self.separator.separate_fractional(
            x_vals, y_vals=y_vals_array, max_cuts=max_cuts, iteration=iteration, node_count=node_count
        )

        # Store cuts for pool re-evaluation at child nodes (active via _evaluate_pool_cuts).
        for cut in cuts:
            # Signature: (node_set, class_name, facet_form, node_i, node_j)
            node_set = getattr(cut, "node_set", set())
            sig = (
                frozenset(node_set) if node_set else frozenset(),
                cut.__class__.__name__,
                getattr(cut, "facet_form", None),
                getattr(cut, "node_i", None),
                getattr(cut, "node_j", None),
            )
            if sig not in self._cut_signatures:
                self._cut_pool.append(cut)
                self._cut_signatures.add(sig)

        # Add cuts as user cuts (not lazy - for LP relaxation strengthening)
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_user(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_user(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        # SAA SIRP separation scaffolds for fractional LP tightening
        if self.params.use_saa and self.scenarios:
            self._separate_stochastic_capacity_cuts(model, x_vals, y_vals_array, is_integer=False)
            self._separate_multi_star_inequalities(model, x_vals, y_vals_array, is_integer=False)
            self._separate_lot_sizing_inequalities(model, x_vals, y_vals_array, is_integer=False)

    def _add_pcsec_lazy(self, model, cut: PCSubtourEliminationCut):
        """
        Add a prize-collecting subtour elimination cut (PC-SEC) as a lazy constraint.

        Lazy constraints are only checked for integer solutions and are essential
        for correctness in the Natural Edge Formulation.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbLazy(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_lazy(self, model, cut: CapacityCut):
        """
        Add a capacity cut as a lazy constraint (integer callback).

        Lazy constraints are only checked for integer solutions and are essential
        for correctness when capacity violations cannot be detected by base constraints.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)

    def _add_pcsec_user(self, model, cut: PCSubtourEliminationCut):
        """
        Add a prize-collecting subtour elimination cut (PC-SEC) as a user cut.

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness (unlike lazy constraints).
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbCut(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbCut(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbCut(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_user(self, model, cut: CapacityCut):
        """
        Add a capacity cut as a user cut (fractional callback).

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbCut(gp.quicksum(edge_vars) >= cut.rhs)

    def _separate_stochastic_capacity_cuts(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """
        Separate Stochastic Capacity Inequalities (SIRP).

        Evaluates scenarios to find subsets where expected demand frequently exceeds capacity,
        adding cuts that force additional vehicles or node shedding.
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            return

        # Find connected components of fractional support graph (edges > 0.1)
        support_edges = [(u, v) for (u, v), val in zip(self.model.edges, x_vals, strict=False) if val > 0.1]
        if NETWORKX_AVAILABLE:
            G = nx.Graph()
            G.add_nodes_from(range(self.model.n_nodes))
            G.add_edges_from(support_edges)
            components = list(nx.connected_components(G))
        else:
            warnings.warn("NetworkX not available, skipping stochastic capacity cuts", stacklevel=2)
            return  # Requires networkx

        num_s = len(self.scenarios)
        avg_demands = {i: sum(scen.get(i, 0.0) for scen in self.scenarios) / num_s for i in self.model.customers}
        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        for comp in components:
            if self.model.depot in comp:
                continue

            S = list(comp)
            if len(S) <= 1:
                continue

            expected_demand = sum(avg_demands.get(i, 0.0) * y_val_dict.get(i, 0.0) for i in S)  # type: ignore[no-matching-overload]

            if expected_demand > 0:
                req_capacity = (2.0 * expected_demand) / self.model.capacity

                delta_S_edges = [(u, v) for u, v in self.model.edges if (u in S) != (v in S)]
                delta_S_val = sum(x_vals[edges_list.index((u, v))] for u, v in delta_S_edges)

                if delta_S_val < req_capacity - 1e-4:
                    edge_vars = [
                        self.x_vars[tuple(sorted(e))]  # type: ignore[index]
                        for e in delta_S_edges
                        if tuple(sorted(e)) in self.x_vars
                    ]
                    y_vars_S = [self.y_vars[i] for i in S if i in self.y_vars]
                    demand_S = [avg_demands.get(i, 0.0) for i in S if i in self.y_vars]

                    if edge_vars and y_vars_S:
                        cut_expr = gp.quicksum(edge_vars) >= (2.0 / self.model.capacity) * gp.quicksum(
                            d * y for d, y in zip(demand_S, y_vars_S, strict=False)
                        )
                        if is_integer:
                            model.cbLazy(cut_expr)
                        else:
                            model.cbCut(cut_expr)
                        self.stats["total_cuts"] += 1

    def _separate_multi_star_inequalities(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """
        Separate Multi-star Inequalities (SIRP).

        Strengthens the routing structure around subsets of correlated high-demand nodes
        across multiple stochastic scenarios.
        """
        if not hasattr(self, "scenarios") or not self.scenarios:
            return

        num_s = len(self.scenarios)
        avg_demands = {i: sum(scen.get(i, 0.0) for scen in self.scenarios) / num_s for i in self.model.customers}
        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        # Identify "nucleus" nodes: heavily visited with significant demand
        for _idx, i in enumerate(self.model.customers):
            if y_val_dict[i] > 0.5 and avg_demands.get(i, 0.0) > self.model.capacity / 3.0:
                star_edges = []
                star_nodes = {i}
                for j in self.model.customers:
                    if i != j:
                        edge = (min(i, j), max(i, j))
                        if edge in self.model.edges:
                            e_idx = edges_list.index(edge)
                            if x_vals[e_idx] > 0.1:
                                star_edges.append(edge)
                                star_nodes.add(j)

                if len(star_nodes) >= 3:
                    expected_hub_demand = sum(avg_demands.get(k, 0.0) * y_val_dict.get(k, 0.0) for k in star_nodes)
                    if expected_hub_demand > self.model.capacity:
                        edge_vars = [self.x_vars[e] for e in star_edges if e in self.x_vars]
                        y_vars_hub = [self.y_vars[k] for k in star_nodes if k in self.y_vars]
                        demand_hub = [avg_demands.get(k, 0.0) for k in star_nodes if k in self.y_vars]

                        if edge_vars and y_vars_hub:
                            bound = gp.quicksum(y for y in y_vars_hub) - (1.0 / self.model.capacity) * gp.quicksum(
                                d * y for d, y in zip(demand_hub, y_vars_hub, strict=False)
                            )
                            cut_expr = gp.quicksum(edge_vars) <= bound
                            if is_integer:
                                model.cbLazy(cut_expr)
                            else:
                                model.cbCut(cut_expr)
                            self.stats["total_cuts"] += 1

    def _separate_lot_sizing_inequalities(self, model, x_vals: np.ndarray, y_vals: np.ndarray, is_integer: bool):
        """
        Separate inequalities derived from Deterministic Lot-Sizing problems.

        Limits combinations of disjoint routes that consistently produce capacity failures
        in extreme scenarios by linking fractional routing capacity to continuous recourse.
        """
        if not hasattr(self, "q_vars") or not self.q_vars or not self.scenarios:
            return

        support_edges = [(u, v) for (u, v), val in zip(self.model.edges, x_vals, strict=False) if val > 0.1]
        if NETWORKX_AVAILABLE:
            G = nx.Graph()
            G.add_nodes_from(range(self.model.n_nodes))
            G.add_edges_from(support_edges)
            components = list(nx.connected_components(G))
        else:
            warnings.warn("NetworkX not available, skipping lot-sizing cuts", stacklevel=2)
            return

        y_val_dict = {i: y_vals[idx] for idx, i in enumerate(self.model.customers)}
        edges_list = list(self.model.edges)

        for s_idx, q_var in self.q_vars.items():
            scenario_demands = self.scenarios[s_idx]
            q_val = model.cbGetSolution(q_var) if is_integer else model.cbGetNodeRel(q_var)

            for comp in components:
                if self.model.depot in comp:
                    continue

                S = list(comp)
                if len(S) <= 1:
                    continue

                scenario_load = sum(scenario_demands.get(i, 0.0) * y_val_dict.get(i, 0.0) for i in S)  # type: ignore[no-matching-overload]

                delta_S_edges = [(u, v) for u, v in self.model.edges if (u in S) != (v in S)]
                delta_S_val = sum(x_vals[edges_list.index((u, v))] for u, v in delta_S_edges)
                provided_capacity = (delta_S_val / 2.0) * self.model.capacity

                if scenario_load - provided_capacity > q_val + 1e-4:
                    edge_vars = [
                        self.x_vars[tuple(sorted(e))]  # type: ignore[index]
                        for e in delta_S_edges
                        if tuple(sorted(e)) in self.x_vars
                    ]
                    y_vars_S = [self.y_vars[i] for i in S if i in self.y_vars]
                    demand_S = [scenario_demands.get(i, 0.0) for i in S if i in self.y_vars]

                    if edge_vars and y_vars_S:
                        cut_expr = q_var >= gp.quicksum(d * y for d, y in zip(demand_S, y_vars_S, strict=False)) - (
                            self.model.capacity / 2.0
                        ) * gp.quicksum(edge_vars)
                        if is_integer:
                            model.cbLazy(cut_expr)
                        else:
                            model.cbCut(cut_expr)
                        self.stats["total_cuts"] += 1

    def _set_start_solution(self, tour: List[int]):
        """Provide a warm start solution to Gurobi."""
        # Set x variables based on tour
        for i, j in self.model.edges:
            self.x_vars[(i, j)].Start = 0.0

        for idx in range(len(tour) - 1):
            edge = tuple(sorted([tour[idx], tour[idx + 1]]))
            if edge in self.x_vars:
                # Increment start value to handle using same edge twice (e.g. [0, 1, 0])
                current_start = getattr(self.x_vars[edge], "Start", 0.0)  # type: ignore[index]
                if np.isnan(current_start):  # Gurobi might initialize with NaN
                    current_start = 0.0
                self.x_vars[edge].Start = current_start + 1.0  # type: ignore[index]

        # Set y variables based on visited nodes
        visited_nodes = set(tour) - {self.model.depot}
        for i in self.model.customers:
            self.y_vars[i].Start = 1.0 if i in visited_nodes else 0.0

    def _extract_solution(self) -> Tuple[List[List[int]], float]:
        """
        Extract the optimal routes from Gurobi solution.
        Handles multiple routes in the aggregate flow formulation.
        """
        assert self.gurobi_model is not None
        if self.gurobi_model.SolCount == 0:
            return [], 0.0

        # Extract active edges into a multiset-like list
        active_edges = []
        for (i, j), var in self.x_vars.items():
            val = round(var.X)
            for _ in range(int(val)):
                active_edges.append(list(sorted([i, j])))

        if not active_edges:
            return [], 0.0

        routes = []
        remaining_edges = active_edges

        # Multi-vehicle route extraction: trace all paths starting from depot
        while True:
            # Find an edge incident to depot
            start_edge = None
            for e in remaining_edges:
                if self.model.depot in e:
                    start_edge = e
                    break

            if start_edge is None:
                break

            # Start a new route
            route = [self.model.depot]
            current = start_edge[1] if start_edge[0] == self.model.depot else start_edge[0]
            remaining_edges.remove(start_edge)
            route.append(current)

            while current != self.model.depot:
                found_next = False
                for e in remaining_edges:
                    if current in e:
                        next_node = e[1] if e[0] == current else e[0]
                        remaining_edges.remove(e)
                        route.append(next_node)
                        current = next_node
                        found_next = True
                        break
                if not found_next:
                    # Should not happen if SECs are working correctly
                    break

            # Remove depots from the route for adapter consistency.
            # Local mapping expects routes as List[List[customers]].
            customers_only = [node for node in route if node != self.model.depot]
            if customers_only:
                routes.append(customers_only)
            elif len(route) == 3 and route[1] != self.model.depot:
                # Case [0, i, 0] where i is a single visit
                routes.append([route[1]])

        # Compute total profit (obj value isMinimized travel_cost - waste_collected)
        profit = -self.gurobi_model.ObjVal

        return routes, profit

    def _pre_optimize_lagrangian(self) -> List[Set[int]]:  # noqa: C901
        """
        Root Node Strengthening via Lagrangian Relaxation.

        Refactored to extract violated Basic GSECs (1-tree cycles) for direct injection.
        """
        if self.params.verbose:
            print("Strengthening Root Node via Lagrangian Relaxation...")

        n = self.model.n_nodes
        # Initial multipliers
        lambda_mult = np.zeros(n)

        # Held-Karp subgradient parameters (Fischetti et al. 1997, Section 5.1)
        hk_mu = 2.0  # scalar, starts at 2, halved on stagnation
        hk_no_improve = 0  # consecutive iterations without LB improvement
        HK_PATIENCE = 50  # halve mu after this many non-improving iterations
        best_upper_bound = float("inf")  # updated if a primal bound is available
        best_lower_bound = -float("inf")

        # Track unique sets S for injection
        unique_violated_sets: List[Set[int]] = []
        seen_sets = set()

        # Max 1000 iterations as per paper
        for _iter_count in range(1000):
            # 1. Update edge weights with multipliers
            modified_costs = {}
            for i, j in self.model.edges:
                cost = self.model.get_edge_cost(i, j)
                modified_costs[(i, j)] = cost - lambda_mult[i] - lambda_mult[j]

            # 2. Solve Minimum Cost K-Tree (Generalized 1-tree for fleet K)
            # We extract all intermediate components (Basic PC-SECs) from Kruskal's
            k_tree_edges, lb, components = self._solve_k_tree(modified_costs)

            # Identify cycles in the K-tree
            cycle = self._find_k_tree_cycle(k_tree_edges)
            if cycle and self.model.depot not in cycle:
                unique_violated_sets.append(cycle)

            # Add all discovered components S
            for s_set in components:
                if self.model.depot not in s_set and len(s_set) >= 2:
                    s_tuple = tuple(sorted(list(s_set)))
                    if s_tuple not in seen_sets:
                        unique_violated_sets.append(s_set)
                        seen_sets.add(s_tuple)

            # Add constant part of Lagrangian: 2 * sum(lambda_i)
            current_lb = float(lb) + 2 * np.sum(lambda_mult)

            # Record whether this iteration improved the lower bound BEFORE
            # updating best_lower_bound, so the stagnation check below is correct.
            improved_this_iter = current_lb > best_lower_bound
            if improved_this_iter:
                best_lower_bound = float(current_lb)

            # 3. Subgradient Update
            degrees = np.zeros(n)
            for i, j in k_tree_edges:
                degrees[i] += 1
                degrees[j] += 1

            # Subgradient for relaxed degree constraints: b - Ax
            # For customers: 2 - degree_i. For depot: 2K - degree_0.
            subgradient = 2.0 - degrees
            subgradient[0] = 2.0 * self.model.num_vehicles - degrees[0]

            if np.linalg.norm(subgradient) < 1e-6:
                break

            # Held-Karp Subgradient Update: t_k = μ * (UB - LB) / ||g||²
            grad_norm_sq = float(np.dot(subgradient, subgradient))
            if grad_norm_sq > 1e-10:
                hk_step = hk_mu * (best_upper_bound - current_lb) / grad_norm_sq
                lambda_mult += hk_step * subgradient
                # Depot multiplier must be >= 0 (inequality constraint, not equality)
                lambda_mult[0] = max(0.0, lambda_mult[0])

            # Halve μ if no improvement for HK_PATIENCE consecutive iterations.
            # Uses improved_this_iter captured before best_lower_bound was updated.
            if improved_this_iter:
                hk_no_improve = 0
            else:
                hk_no_improve += 1
                if hk_no_improve >= HK_PATIENCE:
                    hk_mu /= 2.0
                    hk_no_improve = 0
                    if hk_mu < 1e-6:
                        break  # Multipliers have converged

        # 4. Initialize variable set J using edges with small reduced costs
        for i, j in self.model.edges:
            cost = self.model.get_edge_cost(i, j)
            red_cost = cost - lambda_mult[i] - lambda_mult[j]
            if red_cost < 1e-4:
                self.x_vars[(i, j)].VarHintVal = 1.0

        if self.params.verbose:
            print(f"Lagrangian phase complete. Root LB: {best_lower_bound:.2f}")

        return unique_violated_sets

    def _find_k_tree_cycle(self, edges: List[Tuple[int, int]]) -> Optional[Set[int]]:
        """
        Find a cycle in a K-tree graph using an iterative stack-based DFS.
        Replaces recursive implementation to prevent RecursionError on large graphs.

        Correctness note: each stack frame carries its own copy of the path from
        the start node to the current node. When we encounter a neighbour v that
        already appears in the current path (a back-edge), we have found a cycle
        and return it immediately. Globally-visited nodes are only skipped when
        they do not appear in the current path, preventing false negatives from
        cross-edges between separate DFS branches.
        """
        adj: Dict[int, List[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        globally_visited: Set[int] = set()

        for start_node in range(self.model.n_nodes):
            if start_node in globally_visited:
                continue

            # Stack entries: (current_node, parent_node, path_from_start_to_current)
            # Each frame carries its own path copy so back-edge detection is local
            # to the branch, not polluted by other branches' visits.
            stack = [(start_node, -1, [start_node])]

            while stack:
                u, parent, path = stack.pop()

                # If u was already fully explored in a previous branch of this
                # component and is NOT in the current path, it cannot form a new
                # cycle through the current branch — skip it.
                if u in globally_visited and u not in path:
                    continue

                globally_visited.add(u)

                for v in adj.get(u, []):
                    if v == parent:
                        # Tree edge back to parent — not a cycle
                        continue
                    if v in path:
                        # Back-edge: v is an ancestor in the current DFS path.
                        # Extract the cycle as the subpath from v to u.
                        idx = path.index(v)
                        return set(path[idx:])
                    # Forward edge: extend the path and push to stack.
                    stack.append((v, u, path + [v]))

        # No cycle found in any component
        return None

    def _solve_k_tree(self, costs: Dict[Tuple[int, int], float]) -> Tuple[List[Tuple[int, int]], float, List[Set[int]]]:
        """
        Solve Minimum Cost K-Tree subproblem (Fischetti et al. 1997).
        Generalized from 1-tree to handle fleet size K.

        Returns:
            Tuple of (edges, cost, all_components_from_kruskal).
        """
        n = self.model.n_nodes
        # MST on nodes 1...n-1
        customers = self.model.customers
        edges_to_mst = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                u, v = customers[i], customers[j]
                edges_to_mst.append((u, v, costs[(u, v)]))

        edges_to_mst.sort(key=lambda x: x[2])

        # Kruskal's with component tracking
        parent = list(range(n))
        node_to_comp = {i: {i} for i in range(n)}
        history_components = []

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        mst_edges = []
        mst_cost = 0.0
        for u, v, cost in edges_to_mst:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                mst_edges.append((u, v))
                mst_cost += cost

                # Merge components and store history
                new_comp = node_to_comp[root_u] | node_to_comp[root_v]
                history_components.append(new_comp)

                parent[root_u] = root_v
                node_to_comp[root_v] = new_comp

        # Select depot edges for the K-tree:
        # Always include the 2 cheapest depot edges (minimum required for connectivity),
        # then greedily add edges with strictly negative reduced cost up to 2K total.
        # Edges with non-negative reduced cost are excluded beyond the mandatory 2,
        # as they cannot improve the Lagrangian lower bound.
        k = self.model.num_vehicles
        depot_edges = []
        for j in self.model.customers:
            depot_edges.append((0, j, costs[(0, j)]))
        depot_edges.sort(key=lambda x: x[2])

        # Always select exactly 2*k cheapest depot edges.
        # For k=1 this enforces a valid 1-tree (exactly 2 depot edges).
        selected_depot_edges = depot_edges[: 2 * k]

        k_tree_edges = mst_edges + [(e[0], e[1]) for e in selected_depot_edges]
        k_tree_cost = mst_cost + sum(e[2] for e in selected_depot_edges)

        return k_tree_edges, k_tree_cost, history_components

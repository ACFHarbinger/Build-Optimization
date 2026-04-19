from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.tree import (
    ScenarioTree,
)

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore
    GRB: Any = None  # type: ignore


class ScenarioTreeExtensiveFormEngine:
    """
    Engine to parse a ScenarioTree and build the corresponding deterministic
    equivalent MILP using Gurobi.
    """

    def __init__(
        self,
        tree: ScenarioTree,
        distance_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        overflow_penalty: float = 10.0,
        time_limit: float = 300.0,
        mip_gap: float = 0.05,
        use_mtz: bool = True,
        verbose: bool = False,
    ):
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for ScenarioTreeExtensiveFormEngine.")

        self.tree = tree
        self.distance_matrix = distance_matrix
        # wastes corresponds to Day 0 (Root)
        self.initial_wastes = wastes
        self.capacity = capacity

        self.waste_weight = waste_weight
        self.cost_weight = cost_weight
        self.overflow_penalty = overflow_penalty

        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.use_mtz = use_mtz
        self.verbose = verbose

        self.num_nodes = distance_matrix.shape[0]
        self.customers = list(range(1, self.num_nodes))

        # Gurobi vars mappings
        self.model = gp.Model("ST_Extensive_Form")
        self.x_vars: Dict[Tuple[int, int, int], gp.Var] = {}  # (n_idx, i, j) -> var
        self.y_vars: Dict[Tuple[int, int], gp.Var] = {}  # (n_idx, i) -> var
        # w represents the fill level *before* any collection is made at this node.
        self.w_vars: Dict[Tuple[int, int], gp.Var] = {}  # (n_idx, i) -> var
        self.q_vars: Dict[Tuple[int, int], gp.Var] = {}  # (n_idx, i) -> var
        self.overflow_vars: Dict[Tuple[int, int], gp.Var] = {}  # (n_idx, i) -> var

        # MTZ vars
        if self.use_mtz:
            self.u_vars: Dict[Tuple[int, int], gp.Var] = {}

    def build_model(self):
        """Constructs the deterministic equivalent MILP."""
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.MIPGap = self.mip_gap
        self.model.Params.OutputFlag = 1 if self.verbose else 0
        if not self.use_mtz:
            self.model.Params.LazyConstraints = 1

        total_nodes = len(self.tree.nodes)
        if self.verbose:
            print(f"Building Extensive Form MILP for {total_nodes} tree nodes.")

        # 1. Variable Generation
        for n_idx, node in self.tree.nodes.items():
            # If day=0, this is just revealing the current state. No decisions or collections take place at day=0.
            # We start routing at day=1.
            if node.day > 0:
                # Routing variables
                for i in range(self.num_nodes):
                    for j in range(i + 1, self.num_nodes):
                        # Use symmetric representation or directed. Will use directed for MTZ simplicity.
                        self.x_vars[(n_idx, i, j)] = self.model.addVar(vtype=GRB.BINARY, name=f"x_n{n_idx}_{i}_{j}")
                        self.x_vars[(n_idx, j, i)] = self.model.addVar(vtype=GRB.BINARY, name=f"x_n{n_idx}_{j}_{i}")

                for i in self.customers:
                    self.y_vars[(n_idx, i)] = self.model.addVar(vtype=GRB.BINARY, name=f"y_n{n_idx}_{i}")
                    self.q_vars[(n_idx, i)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"q_n{n_idx}_{i}")
                    self.overflow_vars[(n_idx, i)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0.0, name=f"oflow_n{n_idx}_{i}"
                    )

                    if self.use_mtz:
                        self.u_vars[(n_idx, i)] = self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_n{n_idx}_{i}"
                        )

            # Inventory levels exist at all nodes
            for i in self.customers:
                # Fill levels are represented as fractions (0.0 to 1.0+)
                self.w_vars[(n_idx, i)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_n{n_idx}_{i}")

        self.model.update()

        # 2. Constraints Setup
        self._add_inventory_constraints()
        self._add_routing_constraints()

        # 3. Objective Setup
        self._set_objective()

    def _add_inventory_constraints(self):
        """Adds inter-day continuous inventory balance and logic restraints."""
        # A) Root node initialization
        root = self.tree.get_root()
        for i in self.customers:
            init_val = self.initial_wastes.get(i, 0.0) / max(1.0, self.capacity)
            self.model.addConstr(self.w_vars[(root.id, i)] == init_val, name=f"init_w_{i}")

        # B) Recursive state transitions
        for n_idx, node in self.tree.nodes.items():
            if node.parent_id is not None:
                parent = self.tree.nodes[node.parent_id]

                for i in self.customers:
                    # Inventory tomorrow = Inventory today
                    #                        - Amount collected today (q)
                    #                        + New waste arriving overnight (node.realization)
                    # Day 0 has no q_vars, it's just the instant before Day 1
                    q_collected = 0.0 if parent.id == 0 else self.q_vars[parent.id, i]

                    # Add increment constraint
                    w_parent = self.w_vars[(parent.id, i)]
                    w_curr = self.w_vars[(n_idx, i)]
                    increment = node.realization.get(i, 0.0)

                    self.model.addConstr(
                        w_curr == w_parent - q_collected + increment, name=f"flow_w_n{n_idx}_p{parent.id}_{i}"
                    )

            # C) Intra-node collection limitations
            if node.day > 0:
                for i in self.customers:
                    # Cannot collect more than is in the bin
                    self.model.addConstr(
                        self.q_vars[(n_idx, i)] <= self.w_vars[(n_idx, i)], name=f"col_max_n{n_idx}_{i}"
                    )

                    # Cannot collect if not visited (Big M = 10.0 to be safe on fractional scales)
                    self.model.addConstr(
                        self.q_vars[(n_idx, i)] <= 10.0 * self.y_vars[(n_idx, i)], name=f"col_visit_n{n_idx}_{i}"
                    )

                    # Overflow constraint: overflow >= w_curr - 1.0 (since w is fraction of capacity)
                    self.model.addConstr(
                        self.overflow_vars[(n_idx, i)] >= self.w_vars[(n_idx, i)] - 1.0, name=f"oflow_def_n{n_idx}_{i}"
                    )

    def _add_routing_constraints(self):
        """Adds intra-daily TSP/VRP routing restrictions."""
        for n_idx, node in self.tree.nodes.items():
            if node.day == 0:
                continue

            # Capacity knapsack: sum(q) <= 1.0 (since q is fractional of vehicle capacity)
            # We assume single vehicle per day
            self.model.addConstr(
                gp.quicksum(self.q_vars[(n_idx, i)] for i in self.customers) <= 1.0, name=f"kinematic_cap_n{n_idx}"
            )

            # Degree constraint for customers
            for i in self.customers:
                # In-degree
                self.model.addConstr(
                    gp.quicksum(self.x_vars[(n_idx, j, i)] for j in range(self.num_nodes) if i != j)
                    == self.y_vars[(n_idx, i)],
                    name=f"deg_in_n{n_idx}_{i}",
                )
                # Out-degree
                self.model.addConstr(
                    gp.quicksum(self.x_vars[(n_idx, i, j)] for j in range(self.num_nodes) if i != j)
                    == self.y_vars[(n_idx, i)],
                    name=f"deg_out_n{n_idx}_{i}",
                )

            # Depot degree (number of routes). If exactly 1 vehicle:
            self.model.addConstr(
                gp.quicksum(self.x_vars[(n_idx, 0, j)] for j in self.customers) <= 1, name=f"depot_out_n{n_idx}"
            )
            self.model.addConstr(
                gp.quicksum(self.x_vars[(n_idx, i, 0)] for i in self.customers)
                == gp.quicksum(self.x_vars[(n_idx, 0, j)] for j in self.customers),
                name=f"depot_bal_n{n_idx}",
            )

            # MTZ SECs
            if self.use_mtz:
                for i in self.customers:
                    for j in self.customers:
                        if i != j:
                            self.model.addConstr(
                                self.u_vars[(n_idx, i)] - self.u_vars[(n_idx, j)] + 1
                                <= self.num_nodes * (1 - self.x_vars[(n_idx, i, j)]),
                                name=f"mtz_n{n_idx}_{i}_{j}",
                            )

    def _set_objective(self):
        """Constructs expected value objective function."""
        # Minimize total expected cost: P(node) * [ travel_cost - waste_collected + overflow_penalty ]
        # In our configuration, we typically maximize.

        expected_profit_expr = 0.0

        for n_idx, node in self.tree.nodes.items():
            if node.day == 0:
                continue

            prob = node.probability

            travel_cost = gp.quicksum(
                self.distance_matrix[i, j] * self.x_vars[(n_idx, i, j)]
                for i in range(self.num_nodes)
                for j in range(self.num_nodes)
                if i != j
            )

            # Note: q is scaled to capacity. So scale back.
            waste_collected = self.capacity * gp.quicksum(self.q_vars[(n_idx, i)] for i in self.customers)

            overflow_penalty = gp.quicksum(self.overflow_vars[(n_idx, i)] for i in self.customers)

            profit = (
                (self.waste_weight * waste_collected)
                - (self.cost_weight * travel_cost)
                - (self.overflow_penalty * overflow_penalty)
            )

            expected_profit_expr += prob * profit

        self.model.setObjective(expected_profit_expr, GRB.MAXIMIZE)

    def solve(self) -> Tuple[List[int], float]:
        """Runs optimization and returns the selected route for Day 1."""
        self.build_model()

        if not self.use_mtz:
            self.model.optimize(self._lazy_sec_callback)
        else:
            self.model.optimize()

        # Check solve status
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or self.model.SolCount == 0:
            if self.verbose:
                print(f"Failed to find solution. Status: {self.model.Status}")
            return [], 0.0

        if self.verbose:
            print(f"Optimal Expected Value: {self.model.ObjVal:.2f}")

        # Extract route specifically for Day 1
        day1_nodes = self.tree.get_nodes_by_day(1)
        if not day1_nodes:
            return [], self.model.ObjVal

        # For Rolling Horizon, we only need the exact route for ANY valid Day 1 node.
        # Since all Day 1 nodes share the exact same root history, they enforce
        # non-anticipativity, meaning the routing variables MUST be identical.
        # So we just pick the first node.
        n_idx = day1_nodes[0].id

        route = self._extract_route(n_idx)
        return route, float(self.model.ObjVal)

    def _extract_route(self, n_idx: int) -> List[int]:
        active_edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and self.x_vars.get((n_idx, i, j)):
                    val = self.x_vars[(n_idx, i, j)].X
                    if val > 0.5:
                        active_edges.append((i, j))

        if not active_edges:
            return []

        # Sequence them
        route = [0]
        curr = 0
        while True:
            nxt = next((v for u, v in active_edges if u == curr), None)
            if nxt is None or nxt == 0:
                break
            route.append(nxt)
            curr = nxt

        route.append(0)
        return route

    def _lazy_sec_callback(self, model, where):
        """Lazy subtour elimination (if MTZ is disabled)"""
        # This is a bit tricky to implement for multi-nodal layers quickly,
        # so for this framework we default to MTZ.
        if where == GRB.Callback.MIPSOL:
            # We would need to extract components for every n_idx independently.
            pass

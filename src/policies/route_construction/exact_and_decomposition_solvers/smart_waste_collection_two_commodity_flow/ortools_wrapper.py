"""
Google OR-Tools (MPSolver) implementation of the SWC-TCF algorithm.
Supports Gurobi, SCIP, CPLEX, and HiGHS backends natively.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from ortools.linear_solver import pywraplp


def _run_ortools_tcf_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    solver_id: str = "SCIP",  # Can be 'GUROBI', 'SCIP', 'HIGHS', 'CPLEX'
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """
    Solves the SWC-TCF using Google OR-Tools MPSolver.
    """
    # Initialize the requested solver backend
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        print(f"[ERROR] Could not create OR-Tools solver with backend: {solver_id}")
        return [0, 0], 0.0, 0.0

    # 1. Gurobi Backend
    if solver_id == "GUROBI":
        # Gurobi parameter strings are usually "ParamName Value"
        solver.SetSolverSpecificParametersAsString(f"Seed {seed}")

    # 2. SCIP Backend
    elif solver_id == "SCIP":
        # SCIP parameter strings are usually "path/to/param = value"
        solver.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {seed}")

    # 3. HiGHS Backend (Not natively accessible via string params in older OR-Tools)
    # If using HiGHS, you may need to rely on the C++ API or check the specific
    # OR-Tools version documentation for HiGHS parameter routing.

    solver.SetTimeLimit(time_limit * 1000)  # OR-Tools expects milliseconds

    # 1. Parameter Extraction
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    enchimentos = np.insert(bins, 0, 0.0)
    S_dict = {i: (enchimentos[i] / 100.0) * B * V for i in nodes}

    # Criticos Mapping
    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos_dict = {0: False}
    for i, bin_id in enumerate(pure_binsids, 1):
        criticos_dict[i] = bin_id in mandatory_nodes

    max_dist = 6000
    valid_arcs = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]

    # 2. Variable Definitions
    x = {}  # Arc selection (binary)
    f = {}  # Waste flow (continuous)
    h = {}  # Empty capacity flow (continuous)

    for i, j in valid_arcs:
        x[i, j] = solver.BoolVar(f"x_{i}_{j}")
        f[i, j] = solver.NumVar(0.0, solver.infinity(), f"f_{i}_{j}")
        h[i, j] = solver.NumVar(0.0, solver.infinity(), f"h_{i}_{j}")

    g = {i: solver.BoolVar(f"g_{i}") for i in nodes}

    max_trucks = number_vehicles if number_vehicles > 0 else n_bins
    k_var = solver.IntVar(0, max_trucks, "k_var")

    # 3. Constraints
    # Capacity constraints on arcs
    for i, j in valid_arcs:
        solver.Add(f[i, j] + h[i, j] == Q * x[i, j])

    # Flow balance for Waste & Empty Capacity
    for i in nodes_real:
        # Waste
        inflow_f = solver.Sum(f[j, i] for j in nodes if (j, i) in valid_arcs)
        outflow_f = solver.Sum(f[i, j] for j in nodes if (i, j) in valid_arcs)
        solver.Add(outflow_f - inflow_f == S_dict[i] * g[i])

        # Empty Capacity
        inflow_h = solver.Sum(h[j, i] for j in nodes if (j, i) in valid_arcs)
        outflow_h = solver.Sum(h[i, j] for j in nodes if (i, j) in valid_arcs)
        solver.Add(inflow_h - outflow_h == S_dict[i] * g[i])

    # Depot balance
    solver.Add(
        solver.Sum(f[i, 0] for i in nodes_real if (i, 0) in valid_arcs)
        == solver.Sum(S_dict[i] * g[i] for i in nodes_real)
    )
    solver.Add(solver.Sum(h[0, j] for j in nodes_real if (0, j) in valid_arcs) == Q * k_var)
    solver.Add(solver.Sum(f[0, j] for j in nodes_real if (0, j) in valid_arcs) == 0)

    # Route Continuity & Vehicle Count
    solver.Add(solver.Sum(x[0, j] for j in nodes_real if (0, j) in valid_arcs) == k_var)

    for j in nodes_real:
        # In-degree equals out-degree equals g[j]
        solver.Add(solver.Sum(x[i, j] for i in nodes if (i, j) in valid_arcs) == g[j])
        solver.Add(solver.Sum(x[j, k] for k in nodes if (j, k) in valid_arcs) == g[j])

    # Mandatory & Pre-assignments
    critical_nodes = [i for i in nodes_real if criticos_dict[i]]
    if critical_nodes:
        min_visits = len(critical_nodes) - len(nodes_real) * delta
        solver.Add(solver.Sum(g[i] for i in critical_nodes) >= min_visits)

    for i in nodes_real:
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            solver.Add(g[i] == 1)
        elif enchimentos[i] < 10 and not criticos_dict[i]:
            solver.Add(g[i] == 0)

    # 4. Objective Function
    objective = solver.Objective()
    if dual_values:
        # Reduced cost optimization for B&P integration
        pi_0 = dual_values.get(0, 0.0)
        for i in nodes_real:
            objective.SetCoefficient(g[i], R * S_dict[i] - dual_values.get(i, 0.0))
        for i, j in valid_arcs:
            objective.SetCoefficient(x[i, j], -0.5 * C * distance_matrix[i][j])
        objective.SetCoefficient(k_var, -pi_0)
    else:
        # Standard objective
        for i in nodes_real:
            objective.SetCoefficient(g[i], R * S_dict[i])
        for i, j in valid_arcs:
            objective.SetCoefficient(x[i, j], -0.5 * C * distance_matrix[i][j])
        objective.SetCoefficient(k_var, -Omega)

    objective.SetMaximization()

    # 5. Optimization & Parsing
    status = solver.Solve()

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        id_map = {0: 0}
        for i, bin_id in enumerate(pure_binsids, 1):
            id_map[i] = bin_id

        arcos_ativos = [(i, j) for (i, j) in valid_arcs if x[i, j].solution_value() > 0.5]

        # Route construction
        rotas = []
        visitados = set()
        for _ in range(int(k_var.solution_value())):
            rota = []
            atual = 0
            while True:
                prox = [j for (i, j) in arcos_ativos if i == atual and (i, j) not in visitados]
                if not prox:
                    break
                j = prox[0]
                visitados.add((atual, j))
                rota.append((atual, j))
                atual = j
                if j == 0:
                    break
            if rota:
                rotas.append(rota)

        contentores_coletados = []
        for rota in rotas:
            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = objective.Value()
        cost = sum([x[i, j].solution_value() * distance_matrix[i][j] for i, j in valid_arcs])

        return [0] + contentores_coletados, profit, cost

    else:
        print("[WARN] OR-Tools TCF could not find a feasible solution.")
        return [0, 0], 0.0, 0.0

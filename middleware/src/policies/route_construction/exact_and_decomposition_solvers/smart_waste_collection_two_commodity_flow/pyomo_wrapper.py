"""
Pyomo implementation of the SWC-TCF algorithm.
Serves as the high-level Algebraic Modeling Language (AML) baseline.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray


def _run_pyomo_tcf_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    solver_id: str = "scip",
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], float, float]:
    """Builds and solves the SWC-TCF using Pyomo."""
    # 1. Parameter Extraction
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    n_bins = len(bins)
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    enchimentos = np.insert(bins, 0, 0.0)
    S_dict = {i: (enchimentos[i] / 100.0) * B * V for i in nodes}

    pure_binsids = binsids[1:] if len(binsids) == n_bins + 1 else binsids
    criticos_dict = {0: False}
    for i, bin_id in enumerate(pure_binsids, 1):
        criticos_dict[i] = bin_id in mandatory_nodes

    max_dist = 6000

    # 2. Pyomo Model Initialization
    model = pyo.ConcreteModel(name="SWC_TCF_Pyomo")
    model.V = pyo.Set(initialize=nodes)
    model.V_real = pyo.Set(initialize=nodes_real)

    def valid_arcs_rule(m, i, j):
        return i != j and distance_matrix[i][j] <= max_dist

    model.A = pyo.Set(within=model.V * model.V, filter=valid_arcs_rule)

    # Variables
    model.x = pyo.Var(model.A, within=pyo.Binary)
    model.f = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.h = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.g = pyo.Var(model.V, within=pyo.Binary)

    max_trucks = number_vehicles if number_vehicles > 0 else n_bins
    model.k_var = pyo.Var(within=pyo.Integers, bounds=(0, max_trucks))

    # 3. Constraints
    def cap_match_rule(m, i, j):
        return m.f[i, j] + m.h[i, j] == Q * m.x[i, j]

    model.cap_match = pyo.Constraint(model.A, rule=cap_match_rule)

    def flow_waste_rule(m, i):
        inflow = sum(m.f[j, i] for j in m.V if (j, i) in m.A)
        outflow = sum(m.f[i, j] for j in m.V if (i, j) in m.A)
        return outflow - inflow == S_dict[i] * m.g[i]

    model.flow_waste = pyo.Constraint(model.V_real, rule=flow_waste_rule)

    def flow_empty_rule(m, i):
        inflow = sum(m.h[j, i] for j in m.V if (j, i) in m.A)
        outflow = sum(m.h[i, j] for j in m.V if (i, j) in m.A)
        return inflow - outflow == S_dict[i] * m.g[i]

    model.flow_empty = pyo.Constraint(model.V_real, rule=flow_empty_rule)

    model.depot_waste_in = pyo.Constraint(
        expr=sum(model.f[i, 0] for i in model.V_real if (i, 0) in model.A)
        == sum(S_dict[i] * model.g[i] for i in model.V_real)
    )
    model.depot_empty_out = pyo.Constraint(
        expr=sum(model.h[0, j] for j in model.V_real if (0, j) in model.A) == Q * model.k_var
    )
    model.depot_waste_out = pyo.Constraint(expr=sum(model.f[0, j] for j in model.V_real if (0, j) in model.A) == 0)

    model.vehicle_count = pyo.Constraint(
        expr=sum(model.x[0, j] for j in model.V_real if (0, j) in model.A) == model.k_var
    )

    def route_in_rule(m, j):
        return sum(m.x[i, j] for i in m.V if (i, j) in m.A) == m.g[j]

    model.route_in = pyo.Constraint(model.V_real, rule=route_in_rule)

    def route_out_rule(m, j):
        return sum(m.x[j, k] for k in m.V if (j, k) in m.A) == m.g[j]

    model.route_out = pyo.Constraint(model.V_real, rule=route_out_rule)

    # Mandatory & Pre-assignments
    critical_nodes = [i for i in nodes_real if criticos_dict[i]]
    if critical_nodes:
        min_visits = len(critical_nodes) - len(nodes_real) * delta
        model.mandatory_coverage = pyo.Constraint(expr=sum(model.g[i] for i in critical_nodes) >= min_visits)

    model.forced_visits = pyo.ConstraintList()
    for i in nodes_real:
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            model.forced_visits.add(model.g[i] == 1)
        elif enchimentos[i] < 10 and not criticos_dict[i]:
            model.g[i].setub(0)

    # 4. Objective Function
    def obj_rule(m):
        if dual_values:
            pi_0 = dual_values.get(0, 0.0)
            profit = sum((R * S_dict[i] - dual_values.get(i, 0.0)) * m.g[i] for i in m.V_real)
            cost = 0.5 * C * sum(m.x[i, j] * distance_matrix[i][j] for i, j in m.A)
            return profit - cost - (pi_0 * m.k_var)
        else:
            profit = R * sum(S_dict[i] * m.g[i] for i in m.V_real)
            cost = 0.5 * C * sum(m.x[i, j] * distance_matrix[i][j] for i, j in m.A)
            return profit - cost - (Omega * m.k_var)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # 5. Optimization
    opt = pyo.SolverFactory(solver_id)
    if solver_id == "gurobi":
        opt.options["Seed"] = seed
    elif solver_id == "scip":
        # SCIP uses randomseedshift to offset its default internal seed
        opt.options["randomization/randomseedshift"] = seed
    elif solver_id in ["appsi_highs", "highs"]:
        opt.options["random_seed"] = seed

    if solver_id == "gurobi":
        opt.options["TimeLimit"] = time_limit
    elif solver_id == "scip":
        opt.options["limits/time"] = time_limit

    results = opt.solve(model, tee=False)

    # 6. Parse Results
    if (
        pyo.check_optimal_termination(results) or pyo.check_optimal_termination(results) is False
    ):  # Checks for Feasible too
        id_map = {0: 0}
        for i, bin_id in enumerate(pure_binsids, 1):
            id_map[i] = bin_id

        arcos_ativos = [(i, j) for (i, j) in model.A if pyo.value(model.x[i, j]) > 0.5]

        rotas = []
        visitados = set()
        for _ in range(int(round(pyo.value(model.k_var)))):
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

        profit = pyo.value(model.obj)
        cost = sum([pyo.value(model.x[i, j]) * distance_matrix[i][j] for i, j in model.A])
        return [0] + contentores_coletados, profit, cost

    print(f"[WARN] Pyomo TCF ({solver_id}) could not find a feasible solution.")
    return [0, 0], 0.0, 0.0

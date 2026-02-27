"""
Gurobi Solver for VRPP.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from numpy.typing import NDArray

from logic.src.constants.routing import HEURISTICS_RATIO, MIP_GAP, NODEFILE_START_GB


def _run_gurobi_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    env: Optional[gp.Env],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    binsids: List[int],
    must_go: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
):
    """Solve the Vehicle Routing Problem with Profits."""
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    n_bins = len(bins)
    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais = [(e / 100) * B * V for e in enchimentos]  # Altera os enchimentos de % pra valor real em KG
    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]
    S_dict = {i: pesos_reais[i] for i in nodes}

    criticos_dict = {0: False}
    for i, bin_id in enumerate(binsids, 1):
        criticos_dict[i] = bin_id in must_go

    max_dist = 6000
    pares_viaveis = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]

    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")
    mdl.Params.LogToConsole = 0

    x = mdl.addVars(pares_viaveis, vtype=GRB.BINARY, name="x")
    y = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="y")
    f = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f")
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")
    for i, j in pares_viaveis:
        mdl.addConstr(y[i, j] <= Q * x[i, j])
        mdl.addConstr(f[i, j] <= len(nodes) * x[i, j])

    for i in nodes_real:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y or (j, i) in y) == S_dict[i] * g[i])

    if number_vehicles == 0:
        number_vehicles = len(binsids)

    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    mdl.addConstr(k_var == quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x))
    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    for j in nodes_real:
        if (idx_deposito, j) in x:
            mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x:
            mdl.addConstr(x[j, idx_deposito] <= g[j])

    mdl.addConstr(
        quicksum(g[i] for i in nodes_real if criticos_dict[i])
        >= len([i for i in nodes_real if criticos_dict[i]]) - len(nodes_real) * delta
    )

    for i in nodes_real:
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)
        if enchimentos[i] < 10 and not criticos_dict[i]:
            g[i].UB = 0

    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == quicksum(g[j] for j in nodes_real))
    for j in nodes_real:
        mdl.addConstr(
            quicksum(f[i, j] for i in nodes if (i, j) in f) - quicksum(f[j, k] for k in nodes if (j, k) in f) == g[j]
        )

    mdl.setObjective(
        R * quicksum(S_dict[i] * g[i] for i in nodes_real)
        - 0.5 * C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
        - Omega * k_var,
        GRB.MAXIMIZE,
    )

    mdl.Params.MIPFocus = 1
    mdl.Params.Heuristics = HEURISTICS_RATIO
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2
    mdl.Params.CoverCuts = 2
    mdl.Params.FlowCoverCuts = 2
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = NODEFILE_START_GB
    mdl.setParam("MIPGap", MIP_GAP)
    mdl.Params.TimeLimit = time_limit

    contentores_coletados = []
    profit = 0.0
    cost = 0.0
    mdl.optimize()
    if mdl.SolCount > 0:
        id_map = {0: 0}
        for i, bin_id in enumerate(binsids, 1):
            id_map[i] = bin_id
        arcos_ativos = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].X > 0.5]

        rotas = []
        visitados = set()
        while True:
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
            else:
                break

        for rota in rotas:
            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = float(mdl.ObjVal)
        cost = sum([x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis])
        print(f"[DEBUG][VRPP-Gurobi] Profit: {profit}, Cost: {cost}, Collected: {len(contentores_coletados)}")

    return [0] + contentores_coletados, profit, cost

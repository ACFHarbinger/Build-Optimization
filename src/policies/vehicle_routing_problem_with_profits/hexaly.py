"""
Hexaly (LocalSolver) Solver for VRPP.
"""

from __future__ import annotations

import io
import sys
from typing import Any, Dict, List, Optional, cast

import hexaly.optimizer as hx
import numpy as np
from numpy.typing import NDArray

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _run_hexaly_optimizer(  # noqa: C901
    bins: NDArray[np.float64],
    distancematrix: List[List[float]],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    must_go: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    max_iter_no_improv: int = 10,
    recorder: Optional[PolicyStateRecorder] = None,
):
    """
    Solve the Vehicle Routing Problem with Profits using Hexaly Optimizer (Local Search).
    """
    SCALE = 1000
    Omega, delta, psi = values["Omega"], values["delta"], values["psi"]
    Q, R, B, C, V = values["Q"], values["R"], values["B"], values["C"], values["V"]

    Q_int = int(Q * SCALE)
    n_bins = len(bins)

    if number_vehicles == 0:
        number_vehicles = n_bins

    enchimentos = np.insert(bins, 0, 0.0)
    pesos_reais_float = [(e / 100) * B * V for e in enchimentos]
    pesos_reais_int = [int(w * SCALE) for w in pesos_reais_float]

    dist_matrix_int = [[int(dist * SCALE) for dist in row] for row in distancematrix]

    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]

    must_go_set = set(must_go)

    mandatory = set()
    for i in nodes_real:
        if (i in must_go_set) or (enchimentos[i] >= psi * 100):
            mandatory.add(i)

    forbidden = set()
    for i in nodes_real:
        if enchimentos[i] < 10 and i not in must_go_set:
            forbidden.add(i)

    max_dist_int = 6000 * SCALE
    num_nodes = len(nodes)
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model

        dist_array = model.array(dist_matrix_int)
        weights_array = model.array(pesos_reais_int)

        routes = [model.list(num_nodes) for _ in range(number_vehicles)]

        if number_vehicles > 1:
            for k in range(number_vehicles - 1):
                model.constraint(model.count(routes[k]) >= model.count(routes[k + 1]))

        model.constraint(model.disjoint(routes))
        all_visited = model.union(routes)
        model.constraint(model.contains(all_visited, idx_deposito) == 0)

        for node in forbidden:
            model.constraint(model.contains(all_visited, node) == 0)

        total_profit_int: Any = 0
        total_dist_int: Any = 0
        vehicles_used_expr = model.create_constant(0)
        for k in range(number_vehicles):
            route = routes[k]
            count = model.count(route)

            is_used = model.iif(count > 0, 1, 0)
            vehicles_used_expr += is_used

            route_load = model.sum(
                model.range(0, count),
                model.lambda_function(lambda i: model.at(weights_array, model.at(route, i))),  # noqa: B023
            )
            model.constraint(route_load <= Q_int)
            total_profit_int += route_load

            d_start = model.iif(count > 0, model.at(dist_array, idx_deposito, model.at(route, 0)), 0)
            d_path = model.sum(
                model.range(0, count - 1),
                model.lambda_function(lambda i: model.at(dist_array, model.at(route, i), model.at(route, i + 1))),  # noqa: B023
            )
            d_end = model.iif(
                count > 0,
                model.at(dist_array, model.at(route, count - 1), idx_deposito),
                0,
            )

            route_dist = d_start + d_path + d_end
            total_dist_int += route_dist

            if max_dist_int > 0:
                model.constraint(route_dist <= max_dist_int)

        for node in mandatory:
            model.constraint(model.contains(all_visited, node))

        must_go_list = list(must_go)
        if must_go_list:
            nb_must_go = len(must_go_list)
            visited_critical = model.sum([model.contains(all_visited, node) for node in must_go_list])
            limit = max(0, nb_must_go - int(len(nodes_real) * delta))
            model.constraint(visited_critical >= limit)

        revenue_term = (total_profit_int / SCALE) * R
        dist_cost_term = (total_dist_int / SCALE) * (0.5 * C)
        vehicle_cost_term = vehicles_used_expr * Omega

        obj = revenue_term - dist_cost_term - vehicle_cost_term

        model.maximize(obj)
        model.close()

        optimizer.param.time_limit = time_limit
        optimizer.param.verbosity = 0

        best_obj_val = [None]
        no_improv_iter = [0]
        tick_count = [0]

        def callback(opt, type):
            """Callback for early stopping based on convergence detection."""
            if opt.solution.status == hx.HxSolutionStatus.INFEASIBLE:
                return
            if type == hx.HxCallbackType.TIME_TICKED:
                val = opt.solution.get_value(obj)
                tick_count[0] += 1
                if best_obj_val[0] is None:
                    best_obj_val[0] = val
                    no_improv_iter[0] = 0
                else:
                    relative_improvement = abs(val - best_obj_val[0]) / max(abs(best_obj_val[0]), 1e-10)
                    if relative_improvement > 0.001:
                        best_obj_val[0] = val
                        no_improv_iter[0] = 0
                    else:
                        no_improv_iter[0] += 1
                    if no_improv_iter[0] >= max_iter_no_improv:
                        opt.stop()
                if recorder is not None:
                    recorder.record(
                        tick=tick_count[0],
                        best_obj=float(best_obj_val[0]) if best_obj_val[0] is not None else 0.0,
                        no_improv_count=no_improv_iter[0],
                    )

        optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, callback)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            optimizer.solve()
            sys.stdout = old_stdout

            final_route_flat = [0]
            for k in range(number_vehicles):
                r_vals = list(cast(Any, routes[k]).value)
                if not r_vals:
                    continue
                final_route_flat.extend(r_vals)
                final_route_flat.append(0)

            profit = float(cast(Any, obj).value)
            cost = float(cast(Any, total_dist_int).value) / SCALE
            return final_route_flat, profit, cost
        except Exception:
            sys.stdout = old_stdout
            return [0, 0], 0.0, 0.0

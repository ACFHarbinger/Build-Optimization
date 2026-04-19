r"""
Integer L-Shaped Method (Benders Decomposition) policy package.

Implements the Laporte & Louveaux (1993) Integer L-Shaped Method for solving the
Stochastic Inventory Routing Problem (SIRP / SCWCVRP) as a Two-Stage Stochastic
Integer Program.

Public API
----------
``run_ils``
    Canonical solver entry point.  Instantiates VRPPModel and ILS engine,
    executes the Benders outer loop, and returns the best feasible routing.

Key Classes
-----------
``IntegerLShapedEngine``
    Benders outer loop coordinator.
``MasterProblem``
    Gurobi MILP master problem with SEC lazy callbacks and Benders cut management.
``RecourseEvaluator``
    Analytical expected recourse Q̄(ŷ) and Benders cut generator.
``ScenarioGenerator``
    SAA scenario generation via Gamma-distributed fill-level perturbation.
``ILSBDParams``
    Typed configuration dataclass.

References
----------
Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
stochastic integer programs with complete recourse". Operations Research
Letters, 13(3), 133-142.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .ils_bd_engine import IntegerLShapedEngine
from .master_problem import MasterProblem
from .params import ILSBDParams
from .scenario import ScenarioGenerator
from .subproblem import RecourseEvaluator


def run_ils_bd(
    dist_matrix: np.ndarray,
    sub_wastes: Dict[int, float],
    capacity: float,
    revenue: float,
    cost_unit: float,
    params: ILSBDParams,
    mandatory_indices: Set[int],
    vehicle_limit: int = 1,
    **kwargs: Any,
) -> Tuple[List[List[int]], float]:
    r"""Solve a stochastic VRP instance with the Integer L-Shaped Method.

    This is the canonical entry point for the Benders decomposition solver.
    Instantiates the VRPPModel and ILS engine, runs the Benders outer loop,
    and returns the best feasible routing.

    Args:
        dist_matrix: N×N distance matrix (local indices; 0 = depot).
        sub_wastes: Observed fill levels {local_node_idx: fill_%}.
        capacity: Vehicle payload capacity.
        revenue: Revenue per unit of waste collected.
        cost_unit: Cost per unit of distance traveled.
        params: ILSBDParams configuration for the solver.
        mandatory_indices: Set of local node indices that are mandatory.
        vehicle_limit: Number of vehicles available (defaults to 1).

    Returns:
        Tuple of (routes, profit):
            routes: List of customer-only route lists (depot excluded).
            profit: Deterministic net profit = revenue − travel_cost.
    """
    from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel

    n_nodes = len(dist_matrix)
    vrpp_model = VRPPModel(
        n_nodes=n_nodes,
        cost_matrix=dist_matrix,
        wastes=sub_wastes,
        capacity=capacity,
        revenue_per_kg=revenue,
        cost_per_km=cost_unit,
        mandatory_nodes=mandatory_indices,
    )
    vrpp_model.num_vehicles = max(1, vehicle_limit)

    engine = IntegerLShapedEngine(model=vrpp_model, params=params)
    # Note: run_ils_bd is a deterministic fallback; in multi-period it should be passed a tree.
    # We pass None here to satisfy signature if the engine handles it, or a dummy tree.
    # For now, just fix the keyword error.
    routes, _y_hat, profit, _stats = engine.solve(tree=kwargs.get("tree"))  # type: ignore[arg-type]

    return routes, profit


__all__ = [
    "run_ils_bd",
    "IntegerLShapedEngine",
    "MasterProblem",
    "RecourseEvaluator",
    "ScenarioGenerator",
    "ILSBDParams",
]

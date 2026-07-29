"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Optimizer Interface.

Reference:
    Ramos, T. R. P., Morais, C. S., & Barbosa-Povoa, A. P.
    "The smart waste collection routing problem:
    Alternative operational management approaches", 2018.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .ortools_wrapper import _run_ortools_tcf_optimizer
from .pyomo_wrapper import _run_pyomo_tcf_optimizer


def run_swc_tcf_optimizer(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    values: Dict[str, float],
    binsids: List[int],
    mandatory_nodes: List[int],
    number_vehicles: int = 1,
    time_limit: int = 60,
    framework: str = "ortools",
    optimizer: str = "gurobi",
    seed: int = 42,
    dual_values: Optional[Dict[int, float]] = None,
):
    """
    Solve SWC-TCF using either Google OR-Tools or Pyomo wrappers.
    """
    # Map solver names to the exact casing required by the respective frameworks
    if framework == "ortools":
        # OR-Tools requires uppercase identifiers (e.g., 'GUROBI', 'SCIP', 'HIGHS')
        ortools_backend = optimizer.upper()
        if ortools_backend not in ["GUROBI", "SCIP", "HIGHS", "CPLEX"]:
            raise ValueError(f"Unsupported OR-Tools backend: '{ortools_backend}'")

        return _run_ortools_tcf_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            values=values,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            solver_id=ortools_backend,
            seed=seed,
            dual_values=dual_values,
        )

    elif framework == "pyomo":
        # Pyomo requires lowercase identifiers (e.g., 'gurobi', 'scip', 'appsi_highs')
        pyomo_backend = optimizer.lower()
        if pyomo_backend not in ["gurobi", "scip", "appsi_highs", "highs"]:
            raise ValueError(f"Unsupported Pyomo backend: '{pyomo_backend}'")

        return _run_pyomo_tcf_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            values=values,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            number_vehicles=number_vehicles,
            time_limit=time_limit,
            solver_id="appsi_highs" if pyomo_backend == "highs" else pyomo_backend,
            seed=seed,
            dual_values=dual_values,
        )

    else:
        raise ValueError(f"Unknown optimization framework: '{framework}'. Use 'ortools:<solver>' or 'pyomo:<solver>'.")

"""
VRPP Optimizer Interface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from .gurobi import _run_gurobi_optimizer
from .hexaly import _run_hexaly_optimizer


def run_vrpp_optimizer(
    bins: NDArray[np.float64],
    distance_matrix: List[List[float]],
    param: float,
    media: NDArray[np.float64],
    desviopadrao: NDArray[np.float64],
    values: Dict[str, float],
    binsids: List[int],
    must_go: List[int],
    env: Optional[gp.Env] = None,
    number_vehicles: int = 1,
    time_limit: int = 60,
    optimizer: str = "gurobi",
    max_iter_no_improv: int = 10,
):
    """
    Solve VRPP using either Gurobi or Hexaly optimizer.
    """
    if optimizer == "gurobi":
        return _run_gurobi_optimizer(
            bins,
            distance_matrix,
            env,
            param,
            media,
            desviopadrao,
            values,
            binsids,
            must_go,
            number_vehicles,
            time_limit,
        )
    elif optimizer == "hexaly":
        return _run_hexaly_optimizer(
            bins,
            distance_matrix,
            param,
            media,
            desviopadrao,
            values,
            must_go,
            number_vehicles,
            time_limit,
            max_iter_no_improv,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

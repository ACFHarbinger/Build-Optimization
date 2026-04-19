import copy
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal,
    random_removal,
    shaw_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.recreate_repair import greedy_insertion, greedy_profit_insertion
from logic.src.policies.helpers.operators.recreate_repair.regret import regret_k_insertion, regret_k_profit_insertion

_DESTROY_OPS = {
    "random": random_removal,
    "worst": worst_removal,
    "shaw": shaw_removal,
    "cluster": cluster_removal,
}

_REPAIR_OPS = {
    "greedy": greedy_insertion,
    "greedy_profit": greedy_profit_insertion,
    "regret": regret_k_insertion,
    "regret_profit": regret_k_profit_insertion,
}


def apply_lns(  # noqa: C901
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
    q: Optional[int] = None,
    ruin_fraction: Optional[float] = None,
    destroy_op: str = "random",
    repair_op: str = "greedy",
    repair_k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    **kwargs: Any,
) -> List[List[int]]:
    """
    Single ruin-and-recreate pass.

    Default q follows Chen et al. (2018): q = min(ceil(0.05 * n), 10).
    When `q` or `ruin_fraction` is provided, the default formula is overridden.

    Args:
        routes: Current routes.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        rng: Random number generator.
        q: Specific number of nodes to remove.
        ruin_fraction: Fraction of total nodes to remove.
        destroy_op: Name of the destroy operator.
        repair_op: Name of the repair operator.
        repair_k: Number of neighbors to consider for regret operators.
        mandatory_nodes: Nodes that MUST be in the solution.
        **kwargs: Additional parameters for specific operators (e.g., p for worst).

    Returns:
        List[List[int]]: Modified routes.
    """
    total_nodes = sum(len(r) for r in routes)
    if total_nodes == 0:
        return routes

    # Determine q
    if q is not None:
        effective_q = max(1, min(q, total_nodes))
    elif ruin_fraction is not None:
        effective_q = max(1, min(int(ruin_fraction * total_nodes), total_nodes))
    else:
        effective_q = max(1, min(math.ceil(0.05 * total_nodes), 10))

    # Backward compatibility: "regret_2" / "regret_3_profit" style names
    # are parsed into (base_op, repair_k) pairs.
    if repair_op.startswith("regret_"):
        parts = repair_op.split("_")
        # Parts: ["regret", "<k>"] or ["regret", "<k>", "profit"]
        if len(parts) >= 2 and parts[1].isdigit():
            repair_k = int(parts[1])
            repair_op = "regret_profit" if (len(parts) == 3 and parts[2] == "profit") else "regret"

    # Dispatch destroy
    if destroy_op not in _DESTROY_OPS:
        raise ValueError(f"Unknown destroy_op: {destroy_op!r}")

    destroy_fn = _DESTROY_OPS[destroy_op]
    destroy_kwargs: Dict[str, Any] = {"rng": rng}

    if destroy_op == "worst":
        destroy_kwargs["dist_matrix"] = dist_matrix
        destroy_kwargs["p"] = kwargs.get("p", 1.0)
    elif destroy_op == "shaw":
        destroy_kwargs["dist_matrix"] = dist_matrix
        destroy_kwargs["wastes"] = wastes
    elif destroy_op == "cluster":
        destroy_kwargs["dist_matrix"] = dist_matrix
        # Cluster needs the full list of nodes to consider for proximity
        n_bins = dist_matrix.shape[0] - 1
        destroy_kwargs["nodes"] = list(range(1, n_bins + 1))

    new_routes, removed_nodes = destroy_fn(copy.deepcopy(routes), effective_q, **destroy_kwargs)  # type: ignore[operator]

    # Dispatch repair
    if repair_op not in _REPAIR_OPS:
        raise ValueError(f"Unknown repair_op: {repair_op!r}")

    repair_fn = _REPAIR_OPS[repair_op]
    repair_kwargs = {
        "routes": new_routes,
        "removed_nodes": removed_nodes,
        "dist_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "mandatory_nodes": mandatory_nodes,
    }

    if "profit" in repair_op:
        repair_kwargs["R"] = R
        repair_kwargs["C"] = C

    if "regret" in repair_op:
        repair_kwargs["k"] = repair_k

    if "noise" in kwargs:
        repair_kwargs["noise"] = kwargs["noise"]

    return repair_fn(**repair_kwargs)  # type: ignore[call-arg,operator]

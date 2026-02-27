"""
VRPy engine for Branch-Cut-and-Price module.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from vrpy import VehicleRoutingProblem

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def run_bcp_vrpy(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve CVRP using VRPy (Column Generation / Branch-and-Price).

    Note: VRPy does not natively support Waste-Collecting CVRP via simple
    configuration. This implementation solves standard CVRP for ALL nodes
    present in the wastes dictionary (no node dropping).

    Uses NetworkX DiGraph representation.

    Args:
        dist_matrix: Distance matrix (N x N)
        wastes: Node wastes {node_id: waste_value}
        capacity: Vehicle capacity
        R: Revenue per unit waste (unused in this variant)
        C: Cost per unit distance
        values: Config with 'time_limit' (default: 30)
        mandatory_nodes: Optional list of mandatory node indices (unused).

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
            - routes: List of routes excluding Source/Sink
            - cost: Total objective value from VRPy
    """
    if VehicleRoutingProblem is None:
        logging.error("VRPy not installed or import failed.")
        return [], 0.0

    # Suppress Logs
    logging.getLogger("cspy").setLevel(logging.WARNING)
    logging.getLogger("vrpy").setLevel(logging.WARNING)

    G: nx.DiGraph = nx.DiGraph()  # pyrefly: ignore[bad-assignment]
    n_nodes = len(dist_matrix) - 1

    # Add Nodes
    for i in range(1, n_nodes + 1):
        d = wastes.get(i, 0.0)
        G.add_node(i, waste=d)

    # Source and Sink
    SOURCE = "Source"
    SINK = "Sink"

    # Add Edges
    for i in range(1, n_nodes + 1):
        # Depot to customer
        G.add_edge(SOURCE, i, cost=float(dist_matrix[0][i] * C))  # pyrefly: ignore[bad-argument-type]
        # Customer to depot
        G.add_edge(i, SINK, cost=float(dist_matrix[i][0] * C))  # pyrefly: ignore[bad-argument-type]

    for i in range(1, n_nodes + 1):
        for j in range(1, n_nodes + 1):
            if i != j:
                G.add_edge(i, j, cost=float(dist_matrix[i][j] * C))

    prob = VehicleRoutingProblem(G, load_capacity=capacity)

    time_limit = values.get("time_limit", 30)
    prob.solve(time_limit=time_limit)

    if prob.best_routes:
        # VRPy best_routes is a list of paths (each path is a list of nodes)
        routes = []
        for path in prob.best_routes:
            clean_route = [node for node in path if node not in {SOURCE, SINK}]
            if clean_route:
                routes.append(clean_route)
        if recorder is not None:
            recorder.record(engine="vrpy", n_routes=len(routes), cost=float(prob.best_value), solved=1)
        return routes, float(prob.best_value)
    else:
        if recorder is not None:
            recorder.record(engine="vrpy", n_routes=0, cost=0.0, solved=0)
        return [], 0.0

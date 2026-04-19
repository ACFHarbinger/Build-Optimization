"""
Set-Partitioning Polish Route Improver (Gurobi).

Thin wrapper around operators.intensification.set_partitioning_polish*.
Expects a pre-built route pool from kwargs["route_pool"] or
self.config["route_pool"]. For automatic pool construction see
SetPartitioningRouteImprover.
"""

import logging
from typing import Any, List, Tuple

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.intensification_fixing import (
    set_partitioning_polish,
    set_partitioning_polish_profit,
)

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
)

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp

    _HAS_GUROBI = True
    # Quick license ping
    try:
        _test = gp.Model()
        _test.dispose()
    except gp.GurobiError as e:
        _HAS_GUROBI = False
        logger.warning(
            "set_partitioning_polish: Gurobi license check failed (%s); route improver will no-op.",
            e,
        )
except ImportError:
    _HAS_GUROBI = False
    logger.warning("set_partitioning_polish: gurobipy not installed; route improver will no-op.")


@RouteImproverRegistry.register("set_partitioning_polish")
class SetPartitioningPolishRouteImprover(IRouteImprovement):
    """
    Bare wrapper around set_partitioning_polish*. Requires an externally-
    provided route_pool. Real callers should use SetPartitioningRouteImprover
    unless they have a specific reason to hand-curate the pool.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "SetPartitioningPolishRouteImprover"}

        if not _HAS_GUROBI:
            return tour, {"algorithm": "SetPartitioningPolishRouteImprover"}

        route_pool = kwargs.get("route_pool", self.config.get("route_pool", []))
        if not route_pool:
            # Nothing to polish against — return input unchanged.
            return tour, {"algorithm": "SetPartitioningPolishRouteImprover"}

        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        time_limit = kwargs.get("sp_time_limit", self.config.get("sp_time_limit", 60.0))
        seed = kwargs.get("seed", self.config.get("seed", 42))
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "SetPartitioningPolishRouteImprover"}

            if revenue_kg > 0 or cost_per_km > 0:
                refined = set_partitioning_polish_profit(
                    routes=routes,
                    route_pool=route_pool,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    mandatory_nodes=mandatory_nodes,
                    time_limit=time_limit,
                    seed=seed,
                )
            else:
                refined = set_partitioning_polish(
                    routes=routes,
                    route_pool=route_pool,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    time_limit=time_limit,
                    seed=seed,
                )

            return assemble_tour(refined), {"algorithm": "SetPartitioningPolishRouteImprover"}

        except Exception:
            return tour, {"algorithm": "SetPartitioningPolishRouteImprover"}

"""
Branch-and-Price Route Improver.

Primary path: delegates to the in-house BranchAndPriceSolver with full
feature set (exact RCSPP pricing with ng-routes, edge/Ryan-Foster branching,
column pool management, Lagrangian bounds, two-phase pricing, root-node
RCC separation).

Fallback chain when the in-house solver is unavailable or fails:
    1. In-house BranchAndPriceSolver (primary)
    2. vrpy.VehicleRoutingProblem (secondary)
    3. SetPartitioningRouteImprover with sp_n_perturbations=50 (tertiary)

Wall-clock budget is enforced at the route improver level via bp_time_limit.
Branch-and-price can take seconds to minutes per call; do not put this
inside an inner simulation loop.
"""

import logging
import signal
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
)

logger = logging.getLogger(__name__)

# Primary: in-house solver
try:
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.bp import (
        BranchAndPriceSolver,
    )
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.params import BPParams

    _HAS_INHOUSE_BP = True
except ImportError as e:
    _HAS_INHOUSE_BP = False
    logger.warning(
        "branch_and_price: in-house BranchAndPriceSolver not available (%s); "
        "will fall back to vrpy or set_partitioning.",
        e,
    )

# Secondary: vrpy
try:
    from vrpy import VehicleRoutingProblem

    _HAS_VRPY = True
except ImportError:
    _HAS_VRPY = False


if _HAS_VRPY:

    class ProfitableVRP(VehicleRoutingProblem):
        """Type-safe wrapper for VRPy to satisfy Pyright/Pyrefly static analysis."""

        prize_collection: bool


@contextmanager
def _time_limit(seconds: float):
    """Wall-clock timeout context manager (POSIX only)."""

    def _handler(signum, frame):
        raise TimeoutError(f"branch_and_price exceeded {seconds}s time limit")

    if seconds <= 0:
        yield
        return

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@RouteImproverRegistry.register("branch_and_price")
class BranchAndPriceRouteImprover(IRouteImprovement):
    """
    Branch-and-price route improver.

    Primary path delegates to the in-house BranchAndPriceSolver (exact
    pricing, ng-routes, configurable branching). Falls back to vrpy and
    then to a pool-restricted set-partitioning when primary fails.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "BranchAndPriceRouteImprover"}

        # Problem parameters
        wastes = kwargs.get("wastes", {}) if kwargs.get("wastes") is not None else self.config.get("wastes", {})
        capacity = (
            kwargs.get("capacity", float("inf"))
            if kwargs.get("capacity") is not None
            else self.config.get("capacity", float("inf"))
        )
        cost_per_km = (
            kwargs.get("cost_per_km", 0.0)
            if kwargs.get("cost_per_km") is not None
            else self.config.get("cost_per_km", 0.0)
        )
        revenue_kg = (
            kwargs.get("revenue_kg", 0.0)
            if kwargs.get("revenue_kg") is not None
            else self.config.get("revenue_kg", 0.0)
        )
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config) or []

        # Wall-clock budget
        time_limit = (
            kwargs.get("bp_time_limit", 0.0)
            if kwargs.get("bp_time_limit") is not None
            else self.config.get("bp_time_limit", 120.0)
        )

        dm = to_numpy(distance_matrix)

        try:
            input_routes = split_tour(tour)
            if not input_routes:
                return tour, {"algorithm": "BranchAndPriceRouteImprover"}
            input_cost = tour_distance(input_routes, dm)
        except Exception:
            return tour, {"algorithm": "BranchAndPriceRouteImprover"}

        # Create local kwargs to avoid double-passing explicitly handled arguments
        local_kwargs = kwargs.copy()
        for key in ["wastes", "capacity", "cost_per_km", "revenue_kg", "distance_matrix", "distancesC"]:
            local_kwargs.pop(key, None)

        # ---- Primary: in-house solver ----
        if _HAS_INHOUSE_BP:
            try:
                with _time_limit(time_limit):
                    refined = self._solve_inhouse(
                        input_routes=input_routes,
                        dm=dm,
                        wastes=wastes,
                        capacity=capacity,
                        cost_per_km=cost_per_km,
                        revenue_kg=revenue_kg,
                        mandatory_nodes=list(mandatory_nodes),
                        **local_kwargs,
                    )

                if refined is not None:
                    refined_cost = tour_distance(refined, dm)
                    if refined_cost <= input_cost + 1e-6:
                        return assemble_tour(refined), {"algorithm": "BranchAndPriceRouteImprover"}
                    else:
                        logger.debug(
                            "branch_and_price (in-house): worse than input (%.2f vs %.2f); keeping input.",
                            refined_cost,
                            input_cost,
                        )
            except TimeoutError:
                logger.warning("branch_and_price: in-house solver timed out at %ss", time_limit)
            except Exception as e:
                logger.warning("branch_and_price: in-house solver failed (%s); trying vrpy.", e)

        # ---- Secondary: vrpy ----
        if _HAS_VRPY:
            try:
                with _time_limit(time_limit):
                    refined = self._solve_vrpy(
                        input_routes=input_routes,
                        dm=dm,
                        wastes=wastes,
                        capacity=capacity,
                        cost_per_km=cost_per_km,
                        revenue_kg=revenue_kg,
                        mandatory_nodes=list(mandatory_nodes),
                        **local_kwargs,
                    )

                if refined is not None:
                    refined_cost = tour_distance(refined, dm)
                    if refined_cost <= input_cost + 1e-6:
                        return assemble_tour(refined), {"algorithm": "BranchAndPriceRouteImprover"}
            except Exception as e:
                logger.warning("branch_and_price: vrpy failed (%s); falling back to set_partitioning.", e)

        # ---- Tertiary: set_partitioning fallback ----
        return self._fallback_set_partitioning(tour, **kwargs), {"algorithm": "BranchAndPriceRouteImprover"}

    def _solve_inhouse(
        self,
        input_routes: List[List[int]],
        dm: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        cost_per_km: float,
        revenue_kg: float,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Optional[List[List[int]]]:
        """Delegate to the in-house BranchAndPriceSolver."""
        n_nodes = dm.shape[0] - 1

        # Build BPParams from kwargs with bp_* prefixes, falling back to
        # route improver-appropriate defaults.
        #
        # Note: use_exact_pricing defaults to True here because a route improver
        # called once per episode should use the high-quality pricer, even though
        # the solver's own default is False (for speed inside inner loops).
        max_iterations = (
            kwargs.get("bp_max_iterations", 0)
            if kwargs.get("bp_max_iterations") is not None
            else self.config.get("bp_max_iterations", 100)
        )
        max_routes_per_iteration = (
            kwargs.get("bp_max_routes_per_iteration", 0)
            if kwargs.get("bp_max_routes_per_iteration") is not None
            else self.config.get("bp_max_routes_per_iteration", 10)
        )
        optimality_gap = (
            kwargs.get("bp_optimality_gap", 0.0)
            if kwargs.get("bp_optimality_gap") is not None
            else self.config.get("bp_optimality_gap", 1e-4)
        )
        branching_strategy = (
            kwargs.get("bp_branching_strategy", "")
            if kwargs.get("bp_branching_strategy") is not None
            else self.config.get("bp_branching_strategy", "edge")
        )
        max_branch_nodes = (
            kwargs.get("bp_max_branch_nodes", 0)
            if kwargs.get("bp_max_branch_nodes") is not None
            else self.config.get("bp_max_branch_nodes", 1000)
        )
        use_exact_pricing = (
            kwargs.get("bp_use_exact_pricing", False)
            if kwargs.get("bp_use_exact_pricing") is not None
            else self.config.get("bp_use_exact_pricing", True)
        )
        use_ng_routes = (
            kwargs.get("bp_use_ng_routes", False)
            if kwargs.get("bp_use_ng_routes") is not None
            else self.config.get("bp_use_ng_routes", True)
        )
        ng_neighborhood_size = (
            kwargs.get("bp_ng_neighborhood_size", 0)
            if kwargs.get("bp_ng_neighborhood_size") is not None
            else self.config.get("bp_ng_neighborhood_size", 8)
        )
        tree_search_strategy = (
            kwargs.get("bp_tree_search_strategy", "")
            if kwargs.get("bp_tree_search_strategy") is not None
            else self.config.get("bp_tree_search_strategy", "best_first")
        )
        vehicle_limit = (
            kwargs.get("bp_vehicle_limit", 0)
            if kwargs.get("bp_vehicle_limit") is not None
            else self.config.get("bp_vehicle_limit", 0)
        )
        cleanup_frequency = (
            kwargs.get("bp_cleanup_frequency", 0)
            if kwargs.get("bp_cleanup_frequency") is not None
            else self.config.get("bp_cleanup_frequency", 20)
        )
        cleanup_threshold = (
            kwargs.get("bp_cleanup_threshold", 0.0)
            if kwargs.get("bp_cleanup_threshold") is not None
            else self.config.get("bp_cleanup_threshold", -100.0)
        )
        early_termination_gap = (
            kwargs.get("bp_early_termination_gap", 0.0)
            if kwargs.get("bp_early_termination_gap") is not None
            else self.config.get("bp_early_termination_gap", 1e-3)
        )
        allow_heuristic_ryan_foster = (
            kwargs.get("bp_allow_heuristic_ryan_foster", False)
            if kwargs.get("bp_allow_heuristic_ryan_foster") is not None
            else self.config.get("bp_allow_heuristic_ryan_foster", False)
        )
        params = BPParams(
            max_iterations=max_iterations,
            max_routes_per_iteration=max_routes_per_iteration,
            optimality_gap=optimality_gap,
            branching_strategy=branching_strategy,
            max_branch_nodes=max_branch_nodes,
            use_exact_pricing=use_exact_pricing,
            use_ng_routes=use_ng_routes,
            ng_neighborhood_size=ng_neighborhood_size,
            tree_search_strategy=tree_search_strategy,
            vehicle_limit=vehicle_limit,
            cleanup_frequency=cleanup_frequency,
            cleanup_threshold=cleanup_threshold,
            early_termination_gap=early_termination_gap,
            allow_heuristic_ryan_foster=allow_heuristic_ryan_foster,
        )

        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=dm,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=revenue_kg,
            cost_per_km=cost_per_km,
            mandatory_nodes=set(mandatory_nodes),
            params=params,
        )

        flat_tour, profit, statistics = solver.solve()

        if not flat_tour or len(flat_tour) < 2:
            return None

        logger.info(
            "branch_and_price (in-house): profit=%.2f, iterations=%d, columns=%d, proven_optimal=%s",
            profit,
            statistics.get("num_iterations", 0),
            statistics.get("num_columns_generated", 0),
            statistics.get("proven_optimal", False),
        )

        # Convert flat depot-separated tour to list-of-routes
        routes: List[List[int]] = []
        current: List[int] = []
        for node in flat_tour:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)

        return routes

    def _solve_vrpy(  # noqa: C901
        self,
        input_routes: List[List[int]],
        dm: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        cost_per_km: float,
        revenue_kg: float,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Optional[List[List[int]]]:
        """Delegate to vrpy (secondary fallback path)."""
        time_limit = kwargs.get("bp_time_limit", self.config.get("bp_time_limit", 120.0))
        cspy = kwargs.get("bp_use_cspy", self.config.get("bp_use_cspy", True))

        visited_in_input = {n for r in input_routes for n in r}

        # VRPP mode: expand active set to include profitable candidates,
        # not just visited bins. This fixes the bug in the previous implementation
        # that silently capped the route improver's upside.
        if revenue_kg > 0:
            candidate_nodes = set()
            n_bins = dm.shape[0] - 1
            for bin_id in range(1, n_bins + 1):
                single_cost = (dm[0, bin_id] + dm[bin_id, 0]) * cost_per_km
                single_rev = wastes.get(bin_id, 0.0) * revenue_kg
                if single_rev > single_cost:
                    candidate_nodes.add(bin_id)
            active_nodes = sorted(visited_in_input | set(mandatory_nodes) | candidate_nodes)
        else:
            active_nodes = sorted(visited_in_input | set(mandatory_nodes))

        if not active_nodes:
            return None

        G = nx.DiGraph()
        G.add_node("Source", demand=0)
        G.add_node("Sink", demand=0)
        for n in active_nodes:
            G.add_node(str(n), demand=int(wastes.get(n, 0)))
            G.add_edge("Source", str(n), cost=float(dm[0, n]))
            G.add_edge(str(n), "Sink", cost=float(dm[n, 0]))
        for u in active_nodes:
            for v in active_nodes:
                if u != v:
                    G.add_edge(str(u), str(v), cost=float(dm[u, v]))

        # Apply prize collection logic by modifying edge costs
        if revenue_kg > 0:
            for n in active_nodes:
                node_prize = float(wastes.get(n, 0.0)) * revenue_kg
                node_str = str(n)
                # Reduce the cost of entering this node by the value of its prize
                for predecessor in G.predecessors(node_str):
                    # Check if cost is indexable (handles potential list/array of costs)
                    cost_val = G.edges[predecessor, node_str]["cost"]
                    if isinstance(cost_val, (list, np.ndarray)):
                        for k in range(len(cost_val)):
                            cost_val[k] -= node_prize
                    else:
                        G.edges[predecessor, node_str]["cost"] -= node_prize

        prob = ProfitableVRP(
            G,
            load_capacity=int(capacity) if capacity != float("inf") else None,
        )

        if revenue_kg > 0:
            prob.prize_collection = True

        for m_node in mandatory_nodes:
            if str(m_node) in G.nodes:
                G.nodes[str(m_node)]["required"] = True

        prob.solve(cspy=cspy, time_limit=time_limit, solver="cbc")

        routes_data = prob.best_routes
        if routes_data is None:
            return None

        refined: List[List[int]] = []
        # Safely handle both list (actual behavior) and dict (documented behavior)
        if isinstance(routes_data, list):
            for _, node_list in enumerate(routes_data, start=1):
                stripped = [int(n) for n in node_list if n not in ("Source", "Sink")]
                if stripped:
                    refined.append(stripped)
        elif isinstance(routes_data, dict):
            for _, node_list in routes_data.items():
                stripped = [int(n) for n in node_list if n not in ("Source", "Sink")]
                if stripped:
                    refined.append(stripped)

        return refined

    def _fallback_set_partitioning(self, tour: List[int], **kwargs: Any) -> List[int]:
        """Tertiary fallback to pool-restricted set-partitioning."""
        try:
            from .set_partitioning import SetPartitioningRouteImprover

            kwargs.setdefault("sp_n_perturbations", 50)
            sp = SetPartitioningRouteImprover(config=self.config)
            refined_tour, _ = sp.process(tour, **kwargs)
            return refined_tour
        except Exception as e:
            logger.warning("branch_and_price: all fallbacks exhausted (%s); returning input.", e)
            return tour

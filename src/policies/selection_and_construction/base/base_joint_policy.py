"""
Base Joint Policy Module.

Provides :class:`BaseJointPolicy`, an abstract base class for solvers that
perform mandatory-bin selection **and** route construction in a single
integrated optimisation loop.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.interfaces.context.search_context import (
    ConstructionMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)
from logic.src.interfaces.route_constructor import IRouteConstructor
from logic.src.tracking.viz_mixin import PolicyVizMixin


class BaseJointPolicy(PolicyVizMixin, IRouteConstructor):
    """
    Abstract base class for joint selection-and-construction policies.

    Unlike :class:`~logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`,
    this class does **not** separate the selection and construction phases.
    Instead it exposes a single :meth:`solve_joint` entry point that returns
    both the selected bin IDs and the constructed routes in one call.

    Subclasses must implement :meth:`solve_joint`.

    This class implements :class:`~logic.src.interfaces.route_constructor.IRouteConstructor`
    directly, providing a generic ``execute`` method that extracts the
    needed simulation context into a ``JointSelectionConstructionContext``.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config."""
        # Simple config storage mirroring BaseRoutingPolicy
        self._config = config
        self._seed = getattr(config, "seed", 42) if config is not None else 42

    @property
    def config(self) -> Any:
        """Return the policy configuration."""
        return self._config

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for config parsing."""
        return None

    def _get_config_key(self) -> str:
        """Hydra key (defaulting to class name lower)."""
        return type(self).__name__.lower().replace("policy", "")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """
        Solve mandatory-bin selection and route construction jointly.
        """

    # ------------------------------------------------------------------
    # IRouteConstructor implementation
    # ------------------------------------------------------------------

    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[Any]]:
        """
        Simulation-engine entry point. Extracts context and calls solve_joint.
        """
        # 1. Extract inputs from kwargs (sim context)
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")
        search_ctx = kwargs.get("search_context")
        multi_day_ctx = kwargs.get("multi_day_context")

        # Load area-specific constants if available
        # Mirrors BaseRoutingPolicy._load_area_params
        from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

        area = kwargs.get("area", "generic")
        waste_type = kwargs.get("waste_type", "generic")
        values = load_area_and_waste_type_params(area, waste_type)

        # 2. Build Joint Context
        if bins is not None and distance_matrix is not None:
            # Convert bins object (state) to arrays
            bin_ids = bins.bin_ids if hasattr(bins, "bin_ids") else np.arange(1, len(bins.c) + 1, dtype=np.int32)
            current_fill = bins.c if hasattr(bins, "c") else np.zeros(len(bin_ids))

            # Revenue calculation (Revenue per fill unit)
            bin_density = float(values.get("B", 1.0))
            bin_volume = float(values.get("V", 1.0))
            revenue_kg = float(values.get("R", 0.12))
            cost_per_km = float(values.get("C", 0.05))

            context = JointSelectionConstructionContext(
                bin_ids=bin_ids,
                current_fill=current_fill.copy(),
                distance_matrix=distance_matrix,
                capacity=float(values.get("capacity", 500.0)),
                revenue_kg=revenue_kg,
                cost_per_km=cost_per_km,
                bin_density=bin_density,
                bin_volume=bin_volume,
                max_fill=float(values.get("max_fill", 100.0)),
                n_vehicles=int(kwargs.get("n_vehicles", 1)),
                scenario_tree=kwargs.get("scenario_tree"),
                mandatory_override=kwargs.get("mandatory"),
            )
        else:
            raise ValueError("Missing 'bins' or 'distance_matrix' in execute kwargs.")

        # 3. Call core solver
        selected_bins, routes, profit, cost = self.solve_joint(context)

        # 4. Convert outcomes to 5-tuple
        if routes:
            _, _, subset_indices = self._build_subset_problem(selected_bins, context)
            tour = self._routes_to_flat_tour(routes, subset_indices)
        else:
            tour = [0, 0]

        # Metrics enrichment
        sel_patch: SelectionMetrics = {
            "strategy": type(self).__name__,
            "n_selected": len(selected_bins),
        }
        con_patch: ConstructionMetrics = {
            "algorithm": type(self).__name__,
            "n_mandatory": len(selected_bins),
            "profit": profit,
        }

        if search_ctx is not None:
            out_ctx: Optional[SearchContext] = merge_context(
                search_ctx,
                phase=SearchPhase.CONSTRUCTION,
                selection_metrics=sel_patch,
                construction_metrics=con_patch,
            )
        else:
            out_ctx = SearchContext.initialize(
                selection_metrics=sel_patch,
                metadata={"joint_solver": type(self).__name__},
            )
            out_ctx = merge_context(
                out_ctx,
                phase=SearchPhase.CONSTRUCTION,
                construction_metrics=con_patch,
            )

        return tour, cost, profit, out_ctx, multi_day_ctx

    # ------------------------------------------------------------------
    # Shared utility helpers (Moved from solver implementation)
    # ------------------------------------------------------------------

    def _build_subset_problem(
        self,
        selected_bins: List[int],
        context: JointSelectionConstructionContext,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int]]:
        """Build distance sub-matrix and wastes dict."""
        subset_indices = [0] + list(selected_bins)
        dist_np = np.asarray(context.distance_matrix, dtype=float)
        sub_dist = dist_np[np.ix_(subset_indices, subset_indices)]

        sub_wastes: Dict[int, float] = {}
        for local_i, global_i in enumerate(subset_indices[1:], start=1):
            # global_i is ID, current_fill is array indexed by ID-1
            sub_wastes[local_i] = float(context.current_fill[global_i - 1])

        return sub_dist, sub_wastes, subset_indices

    @staticmethod
    def _routes_to_flat_tour(
        routes: List[List[int]],
        subset_indices: List[int],
    ) -> List[int]:
        """Convert local-index routes to a flat global-ID tour."""
        tour: List[int] = [0]
        for route in routes:
            for node in route:
                tour.append(subset_indices[node])
            tour.append(0)
        if len(tour) == 1:
            tour = [0, 0]
        return tour

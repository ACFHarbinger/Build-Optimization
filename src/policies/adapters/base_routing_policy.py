"""
Base Routing Policy Module.

Provides a template base class for routing policies, extracting common
functionality like parameter loading, subset matrix creation, and tour mapping.

This eliminates code duplication across policy_*.py files.
"""

from abc import abstractmethod
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.interfaces.adapter import IPolicyAdapter
from logic.src.policies.tsp import get_route_cost


def _flatten_raw_config(source: Any) -> Dict[str, Any]:
    """Flatten nested config structures (Hydra custom lists, engine dicts) into a flat dict.

    Handles patterns like:
        {"custom": [{"time_limit": 60}, {"start_temp": 100}]}
        {"ortools": [{"time_limit": 60}]}
    """
    result: Dict[str, Any] = {}
    if isinstance(source, list):
        for item in source:
            result.update(_flatten_raw_config(item))
    elif hasattr(source, "items"):
        for k, v in source.items():
            if k in ("gurobi", "ortools", "hexaly", "custom", "params") and isinstance(v, (dict, list)):
                result.update(_flatten_raw_config(v))
            else:
                result[k] = v
    return result


class BaseRoutingPolicy(IPolicyAdapter):
    """
    Base class for routing policies with common utilities.

    Subclasses implement `_run_solver()` and optionally `_get_config_key()`
    and `_config_class()`.

    The `execute()` method provides a template pattern handling:
        1. Must-go validation
        2. Parameter loading
        3. Subset problem creation
        4. Solver invocation (delegated to subclass)
        5. Tour mapping back to global IDs
        6. Cost computation

    Context Inputs (from kwargs):
        must_go: List of 1-based bin IDs to visit
        bins: Bins state object with `.c` fill levels
        distance_matrix: Full NxN distance matrix
        area: Geographic area name
        waste_type: Waste type (e.g., "plastic")
        config: Policy configuration dict

    Context Outputs:
        Tuple[List[int], float, Any]: (tour, cost, extra_output)

    Attributes:
        config: Optional configuration dataclass for this policy.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config dataclass or raw dict.

        Args:
            config: Dataclass config, raw dict from YAML, or None.
                    If a raw dict is passed and the subclass defines
                    ``_config_class()``, it will be parsed into the
                    typed dataclass automatically.
        """
        if config is not None and isinstance(config, dict):
            self._config = self._build_config(config)
        else:
            self._config = config

    @property
    def config(self) -> Any:
        """Return the policy configuration dataclass."""
        return self._config

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for this policy's config.

        Override in subclasses to enable automatic config parsing.
        Returns None by default (no typed config).
        """
        return None

    @classmethod
    def _build_config(cls, raw_config: Dict[str, Any]) -> Any:
        """Build a typed config dataclass from a raw config dict.

        Extracts the policy-specific section from the raw config (using
        ``_get_config_key()``), flattens nested structures, and constructs
        the typed dataclass (from ``_config_class()``).

        Args:
            raw_config: Raw config dict (may be nested with policy key at top level).

        Returns:
            Typed config dataclass, or None if no config class is defined.
        """
        config_cls = cls._config_class()
        if config_cls is None:
            return None

        # Extract policy-specific section
        config_key = cls._get_config_key(cls)  # type: ignore[arg-type]
        policy_section = raw_config.get(config_key, raw_config)

        # Flatten nested structures (custom lists, engine dicts)
        flat = _flatten_raw_config(policy_section)

        # Filter to only fields the dataclass accepts
        valid_fields = {f.name for f in fields(config_cls)}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}

        return config_cls(**filtered)

    def _get_config_key(self) -> str:
        """
        Return the config key for this policy (e.g., 'alns', 'bcp', 'hgs').

        Override in subclasses.
        """
        return "default"

    def _validate_must_go(self, must_go: Optional[List[int]]) -> Optional[Tuple[List[int], float, Any]]:
        """
        Validate must_go input. Returns early result if empty.

        Args:
            must_go: List of bin IDs to visit.

        Returns:
            Empty tour tuple if no targets, else None (continue processing).
        """
        if not must_go:
            return [0, 0], 0.0, None
        return None

    def _load_area_params(
        self, area: str, waste_type: str, config: Dict[str, Any]
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Load area-specific parameters (Q, R, C) and merge with config overrides.

        Uses the typed config dataclass (``self._config``) when available,
        falling back to dict-based extraction for backward compatibility.

        Args:
            area: Geographic area name.
            waste_type: Waste type string.
            config: Policy configuration dict (from context).

        Returns:
            Tuple of (capacity, revenue, cost_unit, merged_values_dict)
        """
        from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

        Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)

        # Build the policy config dict from the typed config or raw dict
        if self._config is not None and is_dataclass(self._config):
            # Typed config takes priority, then overlay any runtime overrides
            config_key = self._get_config_key()
            runtime_overrides = config.get(config_key, {})
            if not isinstance(runtime_overrides, dict):
                runtime_overrides = (
                    _flatten_raw_config(runtime_overrides) if hasattr(runtime_overrides, "items") else {}
                )
            policy_config = {**asdict(self._config), **runtime_overrides}
        else:
            # Legacy path: extract from raw config dict
            config_key = self._get_config_key()
            policy_section = config.get(config_key, config)
            policy_config = (
                _flatten_raw_config(policy_section)
                if hasattr(policy_section, "items") or isinstance(policy_section, list)
                else {}
            )

        capacity = policy_config.get("capacity", Q)
        revenue_kg = policy_config.get("revenue", R)
        cost_unit = policy_config.get("cost_unit", C)
        density = policy_config.get("density", B)
        bin_volume = policy_config.get("bin_volume", V)

        # Scale revenue to euro per percent for VRPP/Classical solvers
        # revenue_kg (€/kg) * density (kg/L) * bin_volume (L) / 100 = € per 1% fill
        revenue_scaled = revenue_kg * (density * bin_volume / 100.0)

        values: Dict[str, Any] = {
            "Q": capacity,
            "R": revenue_scaled,
            "C": cost_unit,
            "B": density,
            "V": bin_volume,
        }
        # Merge all policy config params into values for solver consumption
        values.update(policy_config)
        return capacity, revenue_scaled, cost_unit, values

    def _create_subset_problem(
        self,
        must_go: List[int],
        distance_matrix: Any,
        bins: Any,
        use_all_bins: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int], List[int]]:
        """
        Create a subset distance matrix and wastes dict for the solver.

        Args:
            must_go: List of 1-based global bin IDs to visit.
            distance_matrix: Full distance matrix (list or ndarray).
            bins: Bins object with `.c` attribute (0-indexed fill levels).
            use_all_bins: If True, include all bins in the problem (VRPP mode).

        Returns:
            Tuple of (sub_dist_matrix, sub_wastes, subset_indices, mandatory_nodes)
                - sub_dist_matrix: Distance matrix for subset.
                - sub_wastes: {local_idx: fill} for local nodes 1..M.
                - subset_indices: Mapping from local to global indices.
                - mandatory_nodes: List of local indices that MUST be visited.
        """
        if use_all_bins:
            # Include ALL bins in the solver problem (VRPP model)
            n_bins = len(bins.c)
            subset_indices = [0] + list(range(1, n_bins + 1))
            # In VRPP mode, mandatory nodes are those in the must_go list.
            # Since subset_indices is [0, 1, 2, ..., N], global ID g maps to local index g.
            mandatory_nodes = list(must_go)
        else:
            # Restrict solver ONLY to must_go bins
            # subset_indices[0] = depot (0), subset_indices[1..M] = must_go bins
            subset_indices = [0] + list(must_go)
            # All nodes in this restricted subset are mandatory
            mandatory_nodes = list(range(1, len(subset_indices)))

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Build local wastes: local index i (1-based) -> fill level
        sub_wastes = {}
        for i in range(1, len(subset_indices)):
            global_idx = subset_indices[i]
            # global_idx is 1-based, bins.c is 0-indexed
            fill = bins.c[global_idx - 1]
            sub_wastes[i] = float(fill)

        return sub_dist_matrix, sub_wastes, subset_indices, mandatory_nodes

    def _map_tour_to_global(self, routes: List[List[int]], subset_indices: List[int]) -> List[int]:
        """
        Map solver routes (local indices) back to global bin IDs.

        Args:
            routes: List of routes from solver, each a list of local node indices.
            subset_indices: Mapping from local index to global index.

        Returns:
            Flat tour list with depot (0) separating routes.
        """
        tour = [0]
        if routes:
            for route in routes:
                for node_idx in route:
                    tour.append(subset_indices[node_idx])
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour

    def _compute_cost(self, distance_matrix: Any, tour: List[int]) -> float:
        """
        Compute total tour cost.

        Args:
            distance_matrix: Full distance matrix.
            tour: Tour as list of node indices.

        Returns:
            Total distance cost.
        """
        return get_route_cost(distance_matrix, tour)

    @abstractmethod
    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run the specific solver for this policy.

        Args:
            sub_dist_matrix: Subset distance matrix.
            sub_wastes: Local wastes dict.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
            values: Merged config values.
            mandatory_nodes: Local indices of nodes that MUST be visited.
            **kwargs: Additional solver-specific arguments.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        pass

    def _log_solver_params(self, values: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Log the resolved solver parameters to the active tracking run.

        Captures the full merged ``values`` dict (area defaults + YAML config +
        runtime overrides) as run params, namespaced under the policy name.
        Only logs once per adapter instance to avoid per-day duplication.

        Args:
            values: Merged solver configuration dict from ``_load_area_params``.
            context: Simulation day context kwargs.
        """
        if getattr(self, "_params_logged", False):
            return
        self._params_logged = True

        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is None:
            return

        policy_name = context.get("policy_name", self._get_config_key())
        sample_id = context.get("sample_id", 0)
        prefix = f"policy_params/{policy_name}/s{sample_id}"

        params: Dict[str, Any] = {}
        for k, v in values.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params[f"{prefix}/{k}"] = v

        if params:
            run.log_params(params)

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the routing policy using template method pattern.

        Args:
            **kwargs: Simulation context.

        Returns:
            Tuple of (tour, cost, extra_output)
        """
        must_go = kwargs.get("must_go", [])

        # 1. Validate
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        # 2. Extract context
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")

        # 3. Load parameters
        capacity, revenue, cost_unit, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)
        # 4. Create subset problem
        # use_all_bins=True (VRPP mode) allows the solver to CONSIDER all bins,
        # but only forcefully visit the must_go list (mandatory_nodes).
        sub_dist_matrix, sub_wastes, subset_indices, mandatory_nodes = self._create_subset_problem(
            must_go,
            distance_matrix,
            bins,
            use_all_bins=bool(values.get("vrpp", True)),
        )

        # 5. Run solver (subclass-specific)
        routes, profit, _ = self._run_solver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes,
            **kwargs,
        )

        # 6. Map back to global IDs
        tour = self._map_tour_to_global(routes, subset_indices)

        # 7. Compute cost
        cost = self._compute_cost(distance_matrix, tour)

        return tour, cost, {"profit": profit}

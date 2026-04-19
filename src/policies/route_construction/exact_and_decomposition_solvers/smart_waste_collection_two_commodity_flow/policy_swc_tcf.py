"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Policy Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SWCTCFConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .dispatcher import run_swc_tcf_optimizer
from .params import SWCTCFParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.SOLVER,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("swc_tcf")
class SWCTCFPolicy(BaseRoutingPolicy):
    """
    Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy adapter.

    This policy implements a mathematical programming approach based on the
    Two-Commodity Flow formulation for the Vehicle Routing Problem. It allows
    for the use of either Gurobi (exact) or Hexaly (local search) as the
    underlying optimization engine.

    Technical Context:
    - Formulation: Uses a flow-based MILP model to ensure subtour elimination
      and capacity enforcement.
    - Dispatcher: Multi-backend dispatcher supporting exact and heuristic engines.

    Reference:
    - Ramos, T. R. P., Morais, C. S., & Barbosa-Povoa, A. P. (2018). "The smart
      waste collection routing problem: Alternative operational management
      approaches". Expert Systems with Applications.
    """

    def __init__(self, config: Optional[Union[SWCTCFConfig, Dict[str, Any]]] = None):
        """Initialize SWC-TCF policy with optional config.

        Args:
            config: SWCTCFConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SWCTCFConfig

    def _get_config_key(self) -> str:
        return "swc_tcf"

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
        Legacy single-day solver fallback.
        SWC-TCF uses a specialized execute() method for historical reasons.
        """
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Smart Waste Collection - Two-Commodity Flow (SWC-TCF) solver logic.

        This method coordinates the execution of the SWC-TCF formulation, which
        uses a flow-based Mixed-Integer Linear Program (MILP) to solve the
        routing problem. It supports multiple backends (Gurobi, Hexaly, Pyomo)
        and is specifically tuned for constraints found in the original Ramos
        et al. (2018) smart waste collection study.

        Args:
            **kwargs: Context dictionary containing:
                - area (str): The operational area (e.g., "Rio Maior").
                - waste_type (str): The type of waste (e.g., "plastic").
                - bins (BCollection): The bin collection object or fill amounts.
                - distance_matrix (np.ndarray): Symmetric distance matrix.
                - mandatory_nodes (List[int]): Bins that MUST be collected.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                A 5-tuple containing:
                - route: The optimized collection route (flat list).
                - cost: Total travel cost calculated based on the route.
                - profit: Total net profit (Total Revenue - Total Cost).
                - search_context: The propagated or initialized search context.
                - multi_day_context: The final multi-day state metadata.
        """
        # 1. Extract context
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")
        mandatory_nodes = kwargs.get("mandatory_nodes", [])
        config = kwargs.get("config", {})

        # 2. Load parameters and merge with config
        _, _, _, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)

        # 3. Handle bins input (WSmart+ simulator passes bins object or amounts)
        amounts = bins
        if bins is not None and hasattr(bins, "c"):
            amounts = bins.c

        # Ensure binsids are present (local 1-based for solver consistency)
        n_bins = len(amounts) if amounts is not None else 0
        binsids = list(range(1, n_bins + 1))

        # 4. Initialize type-safe Params
        params = SWCTCFParams.from_config(self._config or values)
        seed = kwargs.get("seed") if kwargs.get("seed") is not None else params.seed

        # 5. Run optimizer
        dual_values = kwargs.get("dual_values")
        route, profit, cost = run_swc_tcf_optimizer(
            bins=amounts,  # type: ignore[arg-type]
            distance_matrix=distance_matrix,  # type: ignore[arg-type]
            values=values,
            binsids=binsids,
            mandatory_nodes=mandatory_nodes,
            number_vehicles=kwargs.get("number_vehicles", 1),
            time_limit=int(params.time_limit),
            framework=params.framework,
            optimizer=params.engine,
            seed=int(seed) if seed is not None else 42,
            dual_values=dual_values,
        )

        return route, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")

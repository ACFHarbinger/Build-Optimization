"""
Neural Policy Adapter Implementation.

This module provides the NeuralPolicy class, which adapts deep reinforcement
learning models to the common IRouteConstructor interface used by the optimization engine.

Attributes:
    RouteConstructorRegistry: Registry where this policy is automatically registered with key "na".

Example:
    >>> from logic.src.policies.neural_agent.policy_neural import NeuralPolicy
    >>> policy = NeuralPolicy()
    >>> route, cost, _ = policy.execute(model_env=env, model_ls=ls, ...)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.models.policies.selection import get_vectorized_selector
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.tracking.core.run import get_active_run
from logic.src.utils.functions import move_to

from .agent import NeuralAgent
from .params import NeuralParams


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.NEURAL_COMBINATORIAL_OPTIMIZATION,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.GPU_ACCELERATED,
)
@RouteConstructorRegistry.register("na")
class NeuralAgentPolicy(BaseRoutingPolicy):
    """
    Neural Agent Policy wrapper that executes deep reinforcement learning models.

    This class handles the interface between the constructive neural search
    and the local search / simulation environment.

    Attributes:
        None: This class is a stateless wrapper.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize NeuralPolicy."""
        super().__init__(config)
        self._params_logged = False

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Neural Policy by performing inference on a DRL model.

        The neural policy encodes the current environment state (distance matrix,
        bin levels, etc.) and uses a Transformer or GNN-based decoder to
        constructively build routes. It supports Hierarchical RL (HRL) via a
        manager-worker architecture if provided.

        Args:
            **kwargs: Context for neural execution, including:
                - model_env: The RL environment instance.
                - model_ls: Local search and profit variable data.
                - bins: The bin collection state.
                - device: Torch device for inference.
                - fill: Current bin fill levels.
                - dm_tensor: Distance matrix as a Torch tensor.
                - hrl_manager (Optional): HRL manager instance for hierarchical decision-making.
                - search_context (Optional[SearchContext]): Context for tracking search metrics.
                - multi_day_context (Optional[MultiDayContext]): Context for inter-day propagation.
                - mandatory (Optional): Explicit list or mask of mandatory nodes.
                - selector_name (Optional): Name of a vectorized mandatory selector.
                - selector_threshold (Optional): Threshold for the mandatory selector.

        Returns:
            Tuple of:
                - tour: The constructed daily tour (list of node IDs).
                - cost: Total travel cost of the neural tour.
                - profit: Net profit (Revenue - Cost).
                - Optional[SearchContext]: Updated search context.
                - Optional[MultiDayContext]: Updated multi-period context.
        """
        model_env = kwargs["model_env"]
        model_ls = kwargs["model_ls"]
        bins = kwargs["bins"]
        device = kwargs["device"]
        fill = kwargs["fill"]
        dm_tensor = kwargs["dm_tensor"]
        hrl_manager = kwargs.get("hrl_manager")

        # 1. Initialize type-safe Params
        # NeuralPolicy typically receives configuration via kwargs["config"] or initialization
        values = kwargs.get("config", {}).get("na", {})
        params = NeuralParams.from_config(self._config or values)

        agent = NeuralAgent(model_env, seed=kwargs.get("seed", params.seed))
        model_data, graph, profit_vars = model_ls

        # Construct cost weights
        cost_weights = {
            "waste": params.waste_weight,
            "length": params.cost_weight,
            "overflows": params.overflow_penalty,
        }

        # Data preparation
        model_data["waste"] = torch.as_tensor(bins.c, dtype=torch.float32).unsqueeze(0)
        if "fill_history" in model_data:
            model_data["current_fill"] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)
        daily_data = move_to(model_data, device)

        # Handle mandatory selection
        mandatory_mask = self._get_mandatory_mask(kwargs, bins, profit_vars, device)

        tour, cost, output_dict = agent.compute_simulator_day(  # type: ignore[call-arg]
            daily_data,
            graph,
            dm_tensor,
            profit_vars,
            hrl_manager=hrl_manager,
            waste_history=bins.get_level_history(device=device),
            cost_weights=cost_weights,
            mandatory=mandatory_mask,
        )

        # Log parameters
        self._log_params(kwargs, cost_weights)

        # Compute profit: collected revenue - travel cost
        visited = {n for n in tour if n != 0}
        collected_revenue = sum(
            float(bins.c[n - 1]) * profit_vars.get("revenue_kg", 1.0) for n in visited if 1 <= n <= len(bins.c)
        )
        profit = collected_revenue - cost * params.cost_weight

        return tour, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")

    def _log_params(self, context: Dict[str, Any], cost_weights: Dict[str, float]) -> None:
        """Log neural policy parameters to the active tracking run."""
        if self._params_logged:
            return
        self._params_logged = True

        run = get_active_run()
        if run is None:
            return

        policy_name = context.get("policy_name", "na")
        sample_id = context.get("sample_id", 0)
        prefix = f"policy_params/{policy_name}/s{sample_id}"

        params: Dict[str, Any] = {f"{prefix}/{k}": v for k, v in cost_weights.items()}
        params[f"{prefix}/selector_name"] = context.get("selector_name")
        params[f"{prefix}/selector_threshold"] = context.get("selector_threshold", 0.7)

        run.log_params(params)

    def _get_mandatory_mask(
        self,
        kwargs: dict,
        bins: Any,
        profit_vars: Optional[dict],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Compute mandatory mask from kwargs or selector.

        Args:
            kwargs: Execution kwargs containing mandatory or selector config.
            bins: Bins object with current fill levels.
            profit_vars: Profit variables (revenue_kg, bin_capacity, etc.)
            device: Target device for tensors.

        Returns:
            Optional[torch.Tensor]: Boolean mask (1, N+1) where True = must visit.
                                     Includes depot at index 0 (always False).
                                     Returns None if no selection is configured.
        """
        # Check for explicit mandatory
        mandatory = kwargs.get("mandatory")
        if mandatory is not None:
            return self._convert_mandatory_to_mask(mandatory, bins, device)

        # Check for selector-based selection
        selector_name = kwargs.get("selector_name")
        if selector_name is None:
            return None

        selector_threshold = kwargs.get("selector_threshold", 0.7)
        current_day = kwargs.get("current_day", 1)

        # Get current fill levels as tensor
        fill_levels = torch.as_tensor(bins.c, dtype=torch.float32, device=device) / 100.0
        fill_levels = fill_levels.unsqueeze(0)  # (1, N)

        # Prepend depot (fill=0)
        depot_fill = torch.zeros(1, 1, dtype=torch.float32, device=device)
        fill_levels_with_depot = torch.cat([depot_fill, fill_levels], dim=1)  # (1, N+1)

        # Get selector and compute mandatory
        try:
            selector = get_vectorized_selector(selector_name, threshold=selector_threshold)
        except ValueError:
            return None

        # Prepare selector kwargs
        selector_kwargs = {
            "threshold": selector_threshold,
            "current_day": torch.tensor(current_day, device=device),
        }

        # Add accumulation rates if available
        if hasattr(bins, "rate") and bins.rate is not None:
            rates = torch.as_tensor(bins.rate, dtype=torch.float32, device=device) / 100.0
            rates = rates.unsqueeze(0)
            depot_rate = torch.zeros(1, 1, dtype=torch.float32, device=device)
            selector_kwargs["accumulation_rates"] = torch.cat([depot_rate, rates], dim=1)

        # Add profit vars for revenue selector
        if profit_vars is not None:
            selector_kwargs["revenue_kg"] = profit_vars.get("revenue_kg", 1.0)
            selector_kwargs["bin_capacity"] = profit_vars.get("bin_capacity", 1.0)

        mandatory_mask = selector.select(fill_levels_with_depot, **selector_kwargs)
        return mandatory_mask

    def _convert_mandatory_to_mask(
        self,
        mandatory: Any,
        bins: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert mandatory input to a boolean mask tensor.

        Args:
            mandatory: List of bin IDs (1-indexed) or boolean mask tensor.
            bins: Bins object to get size.
            device: Target device.

        Returns:
            torch.Tensor: Boolean mask (1, N+1) including depot.
        """
        num_bins = len(bins.c)

        if isinstance(mandatory, torch.Tensor):
            if mandatory.dtype == torch.bool:
                # Already a boolean mask
                if mandatory.dim() == 1:
                    mandatory = mandatory.unsqueeze(0)
                # Ensure depot is included
                if mandatory.size(1) == num_bins:
                    depot = torch.zeros(1, 1, dtype=torch.bool, device=device)
                    mandatory = torch.cat([depot, mandatory.to(device)], dim=1)
                return mandatory.to(device)
            else:
                # Assume list of indices
                mandatory = mandatory.tolist()

        # Convert list of bin IDs to mask
        if isinstance(mandatory, (list, tuple)):
            mask = torch.zeros(1, num_bins + 1, dtype=torch.bool, device=device)
            for bin_id in mandatory:
                if 1 <= bin_id <= num_bins:
                    mask[0, bin_id] = True
            return mask

        return torch.zeros(1, num_bins + 1, dtype=torch.bool, device=device)

    def _run_solver(
        self,
        tour: List[int],
        cost: float,
        profit: float,
        search_context: SearchContext,
        multi_day_context: Any,
    ) -> Tuple[List[int], float, float, SearchContext, Any]:
        """
        Stub to satisfy BaseRoutingPolicy as NeuralAgentPolicy overrides execute().
        """
        return [], 0.0, 0.0, search_context, multi_day_context

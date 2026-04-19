"""
HNA Policy Adapter for Simulator Integration.

Provides a simulator-callable policy shim that wraps the trained
``HRLIRPModule``. When no checkpoint is available, the policy falls back
to a threshold-based Manager + greedy Worker.

Registry key: ``"hna"``
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from logic.src.configs.policies import HNAPolicyConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.helpers.operators import greedy_insertion
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.learning_algorithms.hierarchical_neural_agent.params import HNAParams


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.CONSTRUCTION,
    PolicyTag.GPU_ACCELERATED,
)
@RouteConstructorRegistry.register("hna")
class HierarchicalNeuralAgentPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Simulator-callable policy adapter for the Hierarchical RL (HRL-IRP) algorithm.

    This policy implements a Manager-Worker architecture for the Inventory
    Routing Problem (IRP):
    - **Manager**: A high-level neural network (or threshold-based fallback)
      that decides which nodes must be visited on the current day to prevent
      future stockouts across the $T$-day horizon.
    - **Worker**: A low-level routing heuristic (Greedy Insertion) that
      constructs efficient vehicle tours to satisfy the Manager's visit
      decisions.

    When a pre-trained ``HRLIRPModule`` checkpoint is provided, the Manager
    network selects mandatory nodes based on current fill levels; otherwise,
    it falls back to a greedy threshold heuristic (nodes with fill % above
    ``greedy_threshold``).

    The policy utilizes the "BaseMultiPeriodRoutingPolicy" template to
    propagate multi-day plans and scenario transitions through the simulator.

    Registry key: ``"hna"``
    """

    def __init__(
        self,
        config: Optional[Union[HNAPolicyConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the HRL-IRP simulator policy."""
        super().__init__(config)
        self.params = HNAParams.from_config(config)
        self._module: Optional[Any] = None  # loaded lazily from checkpoint
        self._module_loaded: bool = False

    @classmethod
    def _config_class(cls) -> Type[HNAPolicyConfig]:
        return HNAPolicyConfig

    def _get_config_key(self) -> str:
        return "hna"

    def _load_module(self) -> None:
        """Lazily load the HRLIRPModule from checkpoint."""
        if self._module_loaded:
            return

        params = self.params
        if params.checkpoint_path is not None:
            try:
                from logic.src.pipeline.rl.meta.hrl_irp import HRLIRPModule

                self._module = HRLIRPModule.load_from_checkpoint(
                    params.checkpoint_path,
                    map_location=params.device,
                )
                self._module.eval()
                if params.verbose:
                    print(f"[HRL-IRP] Loaded checkpoint from {params.checkpoint_path}")
            except Exception as exc:
                if params.verbose:
                    print(f"[HRL-IRP] Checkpoint load failed: {exc}. Using threshold fallback.")
                self._module = None
        else:
            self._module = None

        self._module_loaded = True

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _select_mandatory_nodes(
        self,
        wastes: Dict[int, float],
        locs: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Select mandatory nodes for the current day."""
        params = self.params
        self._load_module()
        if self._module is not None and locs is not None:
            try:
                device = params.device
                nodes = sorted(wastes.keys())
                len(nodes)
                fill_tensor = torch.tensor([wastes[n] for n in nodes], dtype=torch.float32, device=device).unsqueeze(0)
                locs_tensor = torch.tensor(locs, dtype=torch.float32, device=device).unsqueeze(0)
                mgr_obs = torch.cat([locs_tensor.view(1, -1), fill_tensor, torch.zeros(1, 1, device=device)], dim=-1)
                with torch.no_grad():
                    if hasattr(self._module, "manager") and self._module.manager is not None:
                        logits = self._module.manager(mgr_obs)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        probs = torch.sigmoid(logits).squeeze(0)
                        mandatory = [nodes[i] for i, p in enumerate(probs) if p.item() > 0.5]
                        return mandatory

            except Exception:
                pass  # Fall through to threshold

        # Threshold fallback
        return [n for n, fill in wastes.items() if fill >= params.greedy_threshold]

    # ------------------------------------------------------------------
    # Multi-period solver
    # ------------------------------------------------------------------

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Hierarchical Neural Agent (HNA) policy.

        Args:
            problem: Current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan spanning the entire horizon.
                - stats: Execution statistics and checkpoint usage metadata.
        """
        params = HNAParams.from_config(self.config)
        sub_dist_matrix = problem.distance_matrix
        sub_wastes = problem.wastes
        mandatory_base = problem.mandatory
        locs = getattr(problem, "locs", None)
        tree = problem.scenario_tree
        capacity = problem.capacity
        revenue = problem.revenue_per_kg
        cost_unit = problem.cost_per_km

        full_plan: List[List[List[int]]] = []
        total_profit = 0.0
        rolling_wastes = dict(sub_wastes)

        for t in range(problem.horizon):
            # Manager: select node subset for this day
            mandatory_selected = self._select_mandatory_nodes(rolling_wastes, locs)
            # Always include base mandatory nodes
            mandatory_today = sorted(set(mandatory_selected) | set(mandatory_base if t == 0 else []))

            if mandatory_today:
                routes = greedy_insertion(
                    routes=[],
                    removed_nodes=list(mandatory_today),
                    dist_matrix=sub_dist_matrix,
                    wastes=rolling_wastes,
                    capacity=capacity,
                    expand_pool=True,
                )
            else:
                routes = []

            # Compute profit
            routing_cost = 0.0
            for route in routes:
                if not route:
                    continue
                routing_cost += sub_dist_matrix[0][route[0]] * cost_unit
                for i in range(len(route) - 1):
                    routing_cost += sub_dist_matrix[route[i]][route[i + 1]] * cost_unit
                routing_cost += sub_dist_matrix[route[-1]][0] * cost_unit

            day_revenue = sum(rolling_wastes.get(n, 0.0) * revenue for r in routes for n in r)
            day_profit = day_revenue - routing_cost
            total_profit += day_profit

            full_plan.append(routes)

            # Update inventory
            visited = {n for r in routes for n in r}
            for n in rolling_wastes:
                if n in visited:
                    rolling_wastes[n] = 0.0
            # Demand transition (use scenario tree or uniform noise)
            if tree is not None and t + 1 < problem.horizon:
                # Use mean fill rates from ProblemContext for rolling prediction
                fill_rates = problem.fill_rate_means
                for n in rolling_wastes:
                    rolling_wastes[n] = min(rolling_wastes[n] + float(fill_rates[n - 1]), 200.0)

            if params.verbose:
                print(f"[HRL-IRP] Day {t}: profit={day_profit:.2f}, visited={visited}")

        today_route = full_plan[0][0] if full_plan and full_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return (
            sol,
            full_plan,
            {
                "total_profit": total_profit,
                "horizon": problem.horizon,
                "checkpoint_used": params.checkpoint_path is not None and self._module is not None,
            },
        )

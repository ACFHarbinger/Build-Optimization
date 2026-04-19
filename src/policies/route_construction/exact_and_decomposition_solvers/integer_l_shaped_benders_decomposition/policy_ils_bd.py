r"""
Integer L-Shaped (Benders Decomposition) Policy Adapter for Stochastic VRPP.

Adapts the Benders decomposition engine to the Multi-Period Stochastic framework,
handling parameter mapping, recourse evaluation, and non-anticipativity
enforcement within a rolling-horizon context.

Algorithm — Integer L-Shaped Method (Laporte & Louveaux, 1993):
    The problem is formulated as a Two-Stage Stochastic Integer Program (2-SIP):

    1. **Stage 1 (Master Problem)**:
       Binary routing arcs $x_{ij}$ and node-visit flags $y_i$ are decided
       for Day 1 before stochastic demand is revealed. A surrogate variable $\theta$
       approximates the expected recourse cost for the remaining $T-1$ days.

    2. **Stage 2 (Recourse Subproblem)**:
       Evaluates the expected penalty $Q(\mathbf{y})$ incurred by the Stage 1
       decisions across all scenarios in the `ScenarioTree`. This includes
       cumulative overflow penalties and missed revenue over the $T$-day horizon.

Iterative Process:
    - **Optimality Cuts**: If the surrogate $\theta$ is less than the true
      expected recourse $Q(\mathbf{y})$, an L-shaped optimality cut is added to
      the master problem.
    - **Combinatorial Cuts**: Since Stage 1 variables are binary, the engine
      generates powerful combinatorial cuts that are valid for all integer
       solutions, accelerating convergence compared to standard Benders.

Implementation:
    - **Lookahead**: Uses the `ScenarioTree` to compute analytical recourse gradients.
    - **Rolling Horizon**: Re-solves the Master Problem each day, updating $\theta$
      based on latest inventory observations.

Registry key: ``"ils_bd"``

References:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import IntegerLShapedBendersConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .ils_bd_engine import IntegerLShapedEngine
from .params import ILSBDParams


@GlobalRegistry.register(
    PolicyTag.EXACT,
    PolicyTag.DECOMPOSITION,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.MULTI_PERIOD,
)
@RouteConstructorRegistry.register("ils_bd")
class IntegerLShapedPolicy(BaseMultiPeriodRoutingPolicy):
    r"""Two-Stage Stochastic MPVRP via the Integer L-Shaped Method.

    This policy formulates the Stochastic Inventory Routing Problem (SIRP) as a
    two-stage stochastic integer program with explicit inventory balance linking
    constraints.

    Master Problem Formulation:
    .. math::

        \min \quad \sum_{t \in T} \text{RoutingCost}_t(\mathbf{y}) + \theta
        \text{s.t.} \quad I_{it} = I_{i,t-1} + d_{it}(\bar\xi) - q_{it}
        \quad \forall i \in V,\, t \in T

        q_{it} \leq M \cdot y_{it}
        \quad \forall i \in V,\, t \in T

    Where $\theta$ represents the lower bound on the expected recourse cost,
    successively tightened by Integer L-Shaped optimality cuts as the
    search progresses.

    Recourse Evaluation:
    The expected cost of bin stockouts and future routing is evaluated across
    the scenario tree. If the Master Problem solution $(\mathbf{y})$ yields an
    expected cost higher than $\theta$, a new cut is added to the Master.

    Registry key: ``"ils_bd"``

    .. note::
        Requires a Gurobi license (``gurobipy``). Without it, the policy
        will raise an ``ImportError`` during the execution phase.
    """

    def __init__(
        self,
        config: Optional[Union[IntegerLShapedBendersConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ILS policy adapter."""
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[IntegerLShapedBendersConfig]:
        """Return the typed configuration class."""
        return IntegerLShapedBendersConfig

    def _get_config_key(self) -> str:
        """Return the YAML config key."""
        return "ils_bd"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Integer L-Shaped Benders Decomposition solver logic.

        This method solves the Multi-Period Stochastic Integer Routing Problem by
        formulating it as a Two-Stage Stochastic Integer Program (2-SIP). It
        implements the methodology of Laporte & Louveaux (1993), decomposing the
        problem into a Master Problem (deciding Day 0 visits and routing) and a
        Recourse Subproblem (evaluating expected penalties and future costs across
        the Scenario Tree).

        It iteratively adds:
        1. **Optimality Cuts**: When the expected recourse for a given visit pattern
           is higher than the Master Problem's surrogate estimate.
        2. **Feasibility Cuts**: To avoid visit patterns that cannot satisfy
           capacity or flow constraints in any scenario.
        3. **Combinatorial Cuts**: To exploit the binary nature of the visit
           decisions for rapid convergence.

        Args:
            problem: The current ProblemContext containing the state, ScenarioTree,
                and problem constants (capacity, revenue, etc.).
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for the Day 0 routing.
                - full_plan: Collection plan containing Day 0 routes and placeholders
                  for future days evaluated by the engine.
                - stats: Execution statistics, including the number of Benders cuts
                  and Gurobi solver metadata.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("Integer L-Shaped requires a ScenarioTree in ProblemContext.")

        n_nodes = len(problem.distance_matrix)
        vrpp_model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=problem.distance_matrix,
            wastes=problem.wastes,  # Day 0 observed
            capacity=problem.capacity,
            revenue_per_kg=problem.revenue_per_kg,
            cost_per_km=problem.cost_per_km,
            mandatory_nodes=set(problem.mandatory),
        )

        # Merge config
        params = ILSBDParams.from_config(asdict(self.config))

        # Invoke Multi-Period Engine
        engine = IntegerLShapedEngine(model=vrpp_model, params=params)

        # Extract inventory context for multi-period constraints
        init_inv = None
        bin_caps = None
        if params.horizon > 1:
            customers = list(vrpp_model.customers)
            init_inv = np.array([float(problem.wastes.get(i, 0.0)) for i in customers])
            # Assuming 100% capacity for all containers unless specified
            bin_caps = np.full(len(customers), 100.0)

        routes, _y_hat, profit, stats = engine.solve(
            tree=tree,
            initial_inventory=init_inv,
            bin_capacities=bin_caps,
        )

        # Build T-period plan snapshot
        full_plan: List[List[List[int]]] = [[] for _ in range(params.horizon + 1)]
        full_plan[0] = routes

        # Standardized return
        today_route = routes[0] if routes else []
        sol_ctx = SolutionContext.from_problem(problem, today_route)

        return sol_ctx, full_plan, stats
        # Note: If we have an inventory plan, it means the master problem reason about day t > 0
        # But in the inventory-linked version, we don't necessarily have x_{ijt} for t > 0.

"""
Lagrangian Selection Strategy Module.

This module implements a selection strategy based on the Lagrangian relaxation
of the Multiple Knapsack Problem (MKP). It uses the dual variables of the
capacity constraints to compute reduced costs for each bin and selects those
with a positive marginal contribution to the overall objective.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_lagrangian import LagrangianSelection
    >>> strategy = LagrangianSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("lagrangian")
class LagrangianSelection(IMandatorySelectionStrategy):
    """
    Selection strategy based on Lagrangian reduced costs.
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins based on positive Lagrangian reduced cost.

        Args:
            context: SelectionContext with bin properties and economic parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if context.distance_matrix is None:
            raise ValueError("LagrangianSelection requires a distance_matrix.")

        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "LagrangianSelection"})

        # Short-circuit on revenue_kg <= 0 or n_vehicles <= 0
        bin_cap = context.bin_volume * context.bin_density
        mass = (context.current_fill / context.max_fill) * bin_cap
        revenue = mass * context.revenue_kg

        if context.revenue_kg <= 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "LagrangianSelection"})

        # Round trip distance to depot
        dist_to_depot = context.distance_matrix[0, 1:]
        round_trip = 2 * dist_to_depot

        if context.n_vehicles <= 0:
            # Unbounded/no-vehicle-limit interpretation: select all revenue-positive bins
            # that cover their round-trip cost.
            rc = revenue - context.cost_per_km * round_trip
            mandatory_indices = np.nonzero(rc > 0)[0]
            return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
                selection_metrics={"strategy": "LagrangianSelection"}
            )

        # Build the LP relaxation of the Multiple Knapsack Problem
        # Objective: Maximize sum(x_i * r_i) - sum(x_i * cost * dist_i)
        # s.t. sum(x_i * m_i) <= n_vehicles * vehicle_capacity
        #      0 <= x_i <= 1

        c = -(revenue - context.cost_per_km * round_trip)  # scipy minimizes
        A = [mass]
        b = [context.n_vehicles * context.vehicle_capacity]
        bounds = [(0, 1) for _ in range(n_bins)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if not res.success:
            # Fallback to simple revenue check if LP fails
            rc = revenue - context.cost_per_km * round_trip
            mandatory_indices = np.nonzero(rc > 0)[0]
            return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
                selection_metrics={"strategy": "LagrangianSelection"}
            )

        # Read dual (shadow price) on the capacity constraint
        # scipy.optimize.linprog returns duals in 'ineqlin' for 'highs' method.
        # With linprog minimizing c @ x subject to A_ub @ x <= b, the marginal
        # for a binding constraint is non-positive. Since we negated the objective
        # for max->min, the shadow price for the original maximization is:
        # lambda = -marginal.
        lambda_star = 0.0
        if hasattr(res, "ineqlin") and len(res.ineqlin.marginals) > 0:
            lambda_star = max(0.0, -float(res.ineqlin.marginals[0]))

        # NOTE: The K-vehicle LP relaxation collapses to a single aggregated
        # knapsack under identical capacities (sum x_i m_i <= K * Q).

        reduced_costs = (revenue - context.cost_per_km * round_trip) - lambda_star * mass

        mandatory_indices = np.nonzero(reduced_costs > 0)[0]
        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "LagrangianSelection"}
        )

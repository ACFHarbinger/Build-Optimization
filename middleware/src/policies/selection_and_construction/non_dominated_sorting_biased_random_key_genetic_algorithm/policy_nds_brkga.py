"""
NDS-BRKGA Joint Policy.

Orchestrates the full NDS-BRKGA algorithm, resolving the joint mandatory-bin
selection and route construction problem in a single population-based search.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.nds_brkga import NDSBRKGAConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.joint_context import JointSelectionConstructionContext
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.selection_and_construction.base.base_joint_policy import BaseJointPolicy
from logic.src.policies.selection_and_construction.base.registry import JointPolicyRegistry
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    compute_adaptive_thresholds,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives import (
    compute_overflow_risk,
    evaluate_population,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params import (
    NDSBRKGAParams,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population import (
    Population,
)


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.SELECTION,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.STOCHASTIC,
    PolicyTag.SINGLE_PERIOD,
    PolicyTag.ANYTIME,
    PolicyTag.JOINT,
)
@JointPolicyRegistry.register("nds_brkga")
@RouteConstructorRegistry.register("nds_brkga")
class NDSBRKGAPolicy(BaseJointPolicy):
    """
    Non-Dominated Sorting Biased Random-Key Genetic Algorithm Policy.

    Jointly optimises mandatory-bin selection and route construction in a
    single population-based evolutionary loop.
    """

    def __init__(self, config: Optional[Union[NDSBRKGAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return NDSBRKGAConfig

    def _get_config_key(self) -> str:
        return "nds_brkga"

    def solve_joint(
        self,
        context: JointSelectionConstructionContext,
    ) -> Tuple[List[int], List[List[int]], float, float]:
        """
        Run the NDS-BRKGA optimisation loop using inputs from the JointContext.
        """
        params = NDSBRKGAParams.from_config(self._config) if self._config else NDSBRKGAParams()
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], [], 0.0, 0.0

        rng = np.random.default_rng(params.seed)
        start_time = time.monotonic()

        # ----------------------------------------------------------------
        # Phase 0: Pre-computation (Risk & Adaptive Thresholds)
        # ----------------------------------------------------------------
        bin_mass = context.bin_mass_kg()
        overflow_risk = compute_overflow_risk(
            context.current_fill,
            bin_mass,
            context.scenario_tree,
            context.overflow_penalty_frac,
        )
        thresholds = compute_adaptive_thresholds(
            overflow_risk,
            threshold_min=params.selection_threshold_min,
            threshold_max=params.selection_threshold_max,
        )

        # ----------------------------------------------------------------
        # Phase 1: Initialise population
        # ----------------------------------------------------------------
        # Revenue scaled for population logic: R = revenue per 1% fill unit
        R_scaled = context.revenue_scaled()
        costs_scaled = context.cost_per_km

        pop = Population.initialise(
            n_bins=n_bins,
            thresholds=thresholds,
            dist_matrix=context.distance_matrix,
            wastes={i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
            capacity=context.capacity,
            R=R_scaled,
            C=costs_scaled,
            overflow_risk=overflow_risk,
            current_fill=context.current_fill,
            bin_density=context.bin_density,
            bin_volume=context.bin_volume,
            max_fill=context.max_fill,
            overflow_penalty_frac=context.overflow_penalty_frac,
            scenario_tree=context.scenario_tree,
            params=params,
            mandatory_override=context.mandatory_override or [],
        )

        # Track global best
        best_chrom, best_obj = pop.best_chromosome()
        best_neg_profit = best_obj[0]

        # ----------------------------------------------------------------
        # Phase 2: Generational loop
        # ----------------------------------------------------------------
        for _gen in range(params.max_generations):
            elapsed = time.monotonic() - start_time
            if params.time_limit > 0 and elapsed >= params.time_limit:
                break

            new_chromosomes = pop.breed_next_generation(
                n_elite=params.n_elite,
                n_mutants=params.n_mutants,
                bias_elite=params.bias_elite,
                rng=rng,
            )

            new_objectives = evaluate_population(
                new_chromosomes,
                thresholds,
                context.distance_matrix,
                {i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
                context.capacity,
                R_scaled,
                costs_scaled,
                overflow_risk,
                params.overflow_penalty,
                context.mandatory_override or [],
            )

            from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2 import (
                select_elite_nsga2,
            )

            _, new_ranks = select_elite_nsga2(new_objectives, params.n_elite)
            pop = Population(new_chromosomes, new_objectives, new_ranks)

            gen_best_chrom, gen_best_obj = pop.best_chromosome()
            if gen_best_obj[0] < best_neg_profit:
                best_chrom = gen_best_chrom
                best_obj = gen_best_obj
                best_neg_profit = gen_best_obj[0]

        # ----------------------------------------------------------------
        # Phase 3: Decode best
        # ----------------------------------------------------------------
        routes = best_chrom.to_routes(
            thresholds,
            {i: context.current_fill[i - 1] for i in range(1, n_bins + 1)},
            context.capacity,
            context.mandatory_override or [],
        )
        selected_bins = sorted({b for route in routes for b in route})

        # Calculate final profit and cost for the return 4-tuple
        visited = set(selected_bins)
        revenue_total = sum(context.current_fill[b - 1] * R_scaled for b in visited)

        total_dist = 0.0
        dm = context.distance_matrix
        for route in routes:
            if not route:
                continue
            total_dist += dm[0, route[0]]
            for k in range(len(route) - 1):
                total_dist += dm[route[k], route[k + 1]]
            total_dist += dm[route[-1], 0]

        cost_total = total_dist * costs_scaled
        profit_total = revenue_total - cost_total

        return selected_bins, routes, profit_total, cost_total

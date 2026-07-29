"""
Route Improvement Factory Module.

This module implements the Factory pattern for creating route improvement operators.
It handles the instantiation of route improvers based on configuration or names.

Attributes:
    RouteImproverFactory (class): The factory class.

Example:
    >>> from logic.src.policies.helpers.route_improvement.base.factory import RouteImproverFactory
    >>> processors = RouteImproverFactory.create_from_config(config)
"""

from typing import Any, List

from logic.src.interfaces.route_improvement import IRouteImprovement

from .registry import RouteImproverRegistry


class RouteImproverFactory:
    """Factory for creating route improvement strategy instances."""

    @staticmethod
    def create(name: str) -> IRouteImprovement:  # noqa: C901
        """
        Create a route improver instance by name.
        """
        from logic.src.policies.route_improvement.adaptive_ensemble import AdaptiveEnsembleRouteImprover
        from logic.src.policies.route_improvement.adaptive_large_neighborhood_search import (
            AdaptiveLargeNeighborhoodSearchRouteImprover,
        )
        from logic.src.policies.route_improvement.branch_and_price import BranchAndPriceRouteImprover
        from logic.src.policies.route_improvement.cheapest_insertion import CheapestInsertionRouteImprover
        from logic.src.policies.route_improvement.cross_exchange import CrossExchangeRouteImprover
        from logic.src.policies.route_improvement.dp_route_reopt import DPRouteReoptRouteImprover
        from logic.src.policies.route_improvement.fast_tsp import FastTSPRouteImprover
        from logic.src.policies.route_improvement.fix_and_optimize import FixAndOptimizeRouteImprover
        from logic.src.policies.route_improvement.guided_local_search import GuidedLocalSearchRouteImprover
        from logic.src.policies.route_improvement.learned import LearnedRouteImprover
        from logic.src.policies.route_improvement.lkh import LinKernighanHelsgaunRouteImprover
        from logic.src.policies.route_improvement.local_search import ClassicalLocalSearchRouteImprover
        from logic.src.policies.route_improvement.mip_lns import MIPLNSRouteImprover
        from logic.src.policies.route_improvement.multi_phase import MultiPhaseRouteImprover
        from logic.src.policies.route_improvement.neural_selector import NeuralSelectorRouteImprover
        from logic.src.policies.route_improvement.node_exchange_steepest import NodeExchangeSteepestRouteImprover
        from logic.src.policies.route_improvement.or_opt import OrOptRouteImprover
        from logic.src.policies.route_improvement.or_opt_steepest import OrOptSteepestRouteImprover
        from logic.src.policies.route_improvement.path import PathRouteImprover
        from logic.src.policies.route_improvement.profitable_detour import ProfitableDetourRouteImprover
        from logic.src.policies.route_improvement.random_local_search import RandomLocalSearchRouteImprover
        from logic.src.policies.route_improvement.regret_k_insertion import RegretKInsertionRouteImprover
        from logic.src.policies.route_improvement.ruin_recreate import RuinRecreateRouteImprover
        from logic.src.policies.route_improvement.set_partitioning import SetPartitioningRouteImprover
        from logic.src.policies.route_improvement.set_partitioning_polish import SetPartitioningPolishRouteImprover
        from logic.src.policies.route_improvement.simulated_annealing import SimulatedAnnealingRouteImprover
        from logic.src.policies.route_improvement.steepest_two_opt import SteepestTwoOptRouteImprover

        cls = RouteImproverRegistry.get_route_improver_class(name)
        if not cls:
            # Fallback for dynamic/mapped names
            n_lower = name.lower()
            if n_lower in ["fast_tsp", "tsp"]:
                return FastTSPRouteImprover()
            elif n_lower in ["local_search", "classical_local_search", "cls"]:
                return ClassicalLocalSearchRouteImprover(operator_name=n_lower)
            elif n_lower in ["random", "random_local_search", "rls"]:
                return RandomLocalSearchRouteImprover()
            elif n_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                return LinKernighanHelsgaunRouteImprover()
            elif n_lower == "path":
                return PathRouteImprover()
            elif n_lower == "or_opt":
                return OrOptRouteImprover()
            elif n_lower == "cross_exchange":
                return CrossExchangeRouteImprover()
            elif n_lower == "guided_local_search":
                return GuidedLocalSearchRouteImprover()
            elif n_lower == "simulated_annealing":
                return SimulatedAnnealingRouteImprover()
            elif n_lower == "cheapest_insertion":
                return CheapestInsertionRouteImprover()
            elif n_lower == "regret_k_insertion":
                return RegretKInsertionRouteImprover()
            elif n_lower == "profitable_detour":
                return ProfitableDetourRouteImprover()
            elif n_lower == "ruin_recreate":
                return RuinRecreateRouteImprover()
            elif n_lower == "adaptive_large_neighborhood_search":
                return AdaptiveLargeNeighborhoodSearchRouteImprover()
            elif n_lower == "multi_phase":
                return MultiPhaseRouteImprover()
            elif n_lower == "branch_and_price":
                return BranchAndPriceRouteImprover()
            elif n_lower in ["dp", "dp_route_reopt"]:
                return DPRouteReoptRouteImprover()
            elif n_lower == "fix_and_optimize":
                return FixAndOptimizeRouteImprover()
            elif n_lower == "learned":
                return LearnedRouteImprover()
            elif n_lower == "node_exchange_steepest":
                return NodeExchangeSteepestRouteImprover()
            elif n_lower == "or_opt_steepest":
                return OrOptSteepestRouteImprover()
            elif n_lower == "set_partitioning":
                return SetPartitioningRouteImprover()
            elif n_lower == "set_partitioning_polish":
                return SetPartitioningPolishRouteImprover()
            elif n_lower == "steepest_two_opt":
                return SteepestTwoOptRouteImprover()
            elif n_lower == "mip_lns":
                return MIPLNSRouteImprover()
            elif n_lower == "neural_selector":
                return NeuralSelectorRouteImprover(config=None)
            elif n_lower == "adaptive_ensemble":
                return AdaptiveEnsembleRouteImprover()

            raise ValueError(f"Unknown route improver: {name}")
        return cls()

    @classmethod
    def create_from_config(cls, config: Any) -> List[IRouteImprovement]:  # noqa: C901
        """
        Create a list of route improver instances from a RouteImprovingConfig object.

        Args:
            config: RouteImprovingConfig instance.
        """
        from logic.src.policies.route_improvement.adaptive_ensemble import AdaptiveEnsembleRouteImprover
        from logic.src.policies.route_improvement.adaptive_large_neighborhood_search import (
            AdaptiveLargeNeighborhoodSearchRouteImprover,
        )
        from logic.src.policies.route_improvement.branch_and_price import BranchAndPriceRouteImprover
        from logic.src.policies.route_improvement.cheapest_insertion import CheapestInsertionRouteImprover
        from logic.src.policies.route_improvement.cross_exchange import CrossExchangeRouteImprover
        from logic.src.policies.route_improvement.dp_route_reopt import DPRouteReoptRouteImprover
        from logic.src.policies.route_improvement.fast_tsp import FastTSPRouteImprover
        from logic.src.policies.route_improvement.fix_and_optimize import FixAndOptimizeRouteImprover
        from logic.src.policies.route_improvement.guided_local_search import GuidedLocalSearchRouteImprover
        from logic.src.policies.route_improvement.learned import LearnedRouteImprover
        from logic.src.policies.route_improvement.lkh import LinKernighanHelsgaunRouteImprover
        from logic.src.policies.route_improvement.local_search import ClassicalLocalSearchRouteImprover
        from logic.src.policies.route_improvement.mip_lns import MIPLNSRouteImprover
        from logic.src.policies.route_improvement.multi_phase import MultiPhaseRouteImprover
        from logic.src.policies.route_improvement.neural_selector import NeuralSelectorRouteImprover
        from logic.src.policies.route_improvement.node_exchange_steepest import NodeExchangeSteepestRouteImprover
        from logic.src.policies.route_improvement.or_opt import OrOptRouteImprover
        from logic.src.policies.route_improvement.or_opt_steepest import OrOptSteepestRouteImprover
        from logic.src.policies.route_improvement.path import PathRouteImprover
        from logic.src.policies.route_improvement.profitable_detour import ProfitableDetourRouteImprover
        from logic.src.policies.route_improvement.random_local_search import RandomLocalSearchRouteImprover
        from logic.src.policies.route_improvement.regret_k_insertion import RegretKInsertionRouteImprover
        from logic.src.policies.route_improvement.ruin_recreate import RuinRecreateRouteImprover
        from logic.src.policies.route_improvement.set_partitioning import SetPartitioningRouteImprover
        from logic.src.policies.route_improvement.set_partitioning_polish import SetPartitioningPolishRouteImprover
        from logic.src.policies.route_improvement.simulated_annealing import SimulatedAnnealingRouteImprover
        from logic.src.policies.route_improvement.steepest_two_opt import SteepestTwoOptRouteImprover

        processors: List[IRouteImprovement] = []
        if not config.methods:
            return processors

        for method in config.methods:
            method_lower = method.lower()
            processor: IRouteImprovement

            # Create route improver with method-specific config parameters
            if method_lower in ["fast_tsp", "tsp"]:
                processor = FastTSPRouteImprover(
                    time_limit=config.fast_tsp.time_limit,
                    seed=config.fast_tsp.seed,
                )
            elif method_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                processor = LinKernighanHelsgaunRouteImprover(
                    max_iterations=config.lkh.max_iterations,
                    time_limit=config.lkh.time_limit,
                    seed=config.lkh.seed,
                )
            elif method_lower in ["local_search", "classical_local_search", "cls"]:
                processor = ClassicalLocalSearchRouteImprover(
                    ls_operator=config.local_search.ls_operator,
                    iterations=config.local_search.iterations,
                    time_limit=config.local_search.time_limit,
                    seed=config.local_search.seed,
                )
            elif method_lower in ["random", "random_local_search", "rls"]:
                processor = RandomLocalSearchRouteImprover(
                    iterations=config.random_local_search.iterations,
                    params=config.random_local_search.params,
                    time_limit=config.random_local_search.time_limit,
                    seed=config.random_local_search.seed,
                )
            elif method_lower == "path":
                processor = PathRouteImprover(
                    vehicle_capacity=config.path.vehicle_capacity,
                )
            elif method_lower == "or_opt":
                processor = OrOptRouteImprover(
                    chain_len=config.or_opt.chain_len,
                    iterations=config.or_opt.iterations,
                    seed=config.or_opt.seed,
                )
            elif method_lower == "cross_exchange":
                processor = CrossExchangeRouteImprover(
                    cross_exchange_max_segment_len=config.cross_exchange.cross_exchange_max_segment_len,
                    iterations=config.cross_exchange.iterations,
                    seed=config.cross_exchange.seed,
                )
            elif method_lower == "guided_local_search":
                processor = GuidedLocalSearchRouteImprover(
                    gls_iterations=config.guided_local_search.gls_iterations,
                    gls_inner_iterations=config.guided_local_search.gls_inner_iterations,
                    gls_lambda_factor=config.guided_local_search.gls_lambda_factor,
                    gls_base_operator=config.guided_local_search.gls_base_operator,
                    seed=config.guided_local_search.seed,
                )
            elif method_lower == "simulated_annealing":
                # Extract Boltzmann params if present in acceptance config
                t_init = 10.0
                alpha = 0.999
                if config.simulated_annealing.acceptance.method == "bmc":
                    bmc_cfg = config.simulated_annealing.acceptance.params
                    if bmc_cfg:
                        t_init = getattr(bmc_cfg, "initial_temp", t_init)
                        alpha = getattr(bmc_cfg, "alpha", alpha)

                processor = SimulatedAnnealingRouteImprover(
                    sa_iterations=config.simulated_annealing.iterations,
                    sa_t_init=t_init,
                    sa_cooling=alpha,
                    params=config.simulated_annealing.params,
                    seed=config.simulated_annealing.seed,
                )
            elif method_lower == "cheapest_insertion":
                processor = CheapestInsertionRouteImprover(
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "regret_k_insertion":
                processor = RegretKInsertionRouteImprover(
                    regret_k=config.insertion.regret_k,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "profitable_detour":
                processor = ProfitableDetourRouteImprover(
                    detour_epsilon=config.insertion.detour_epsilon,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "ruin_recreate":
                processor = RuinRecreateRouteImprover(
                    lns_iterations=config.ruin_recreate.lns_iterations,
                    ruin_fraction=config.ruin_recreate.ruin_fraction,
                    lns_acceptance=config.ruin_recreate.acceptance.method,
                    repair_k=config.ruin_recreate.repair_k,
                    cost_per_km=config.ruin_recreate.cost_per_km,
                    revenue_kg=config.ruin_recreate.revenue_kg,
                    seed=config.ruin_recreate.seed,
                )
            elif method_lower == "adaptive_large_neighborhood_search":
                processor = AdaptiveLargeNeighborhoodSearchRouteImprover(
                    alns_iterations=config.adaptive_lns.alns_iterations,
                    ruin_fraction=config.adaptive_lns.ruin_fraction,
                    alns_bandit_warm_start_path=config.adaptive_lns.alns_bandit_warm_start_path,
                    cost_per_km=config.adaptive_lns.cost_per_km,
                    revenue_kg=config.adaptive_lns.revenue_kg,
                    seed=config.adaptive_lns.seed,
                )
            elif method_lower == "multi_phase":
                processor = MultiPhaseRouteImprover(
                    phases=config.multi_phase.phases,
                    seed=config.multi_phase.seed,
                )
            elif method_lower == "branch_and_price":
                processor = BranchAndPriceRouteImprover(
                    bp_max_iterations=config.branch_and_price.bp_max_iterations,
                    bp_max_routes_per_iteration=config.branch_and_price.bp_max_routes_per_iteration,
                    bp_optimality_gap=config.branch_and_price.bp_optimality_gap,
                    bp_branching_strategy=config.branch_and_price.bp_branching_strategy,
                    bp_max_branch_nodes=config.branch_and_price.bp_max_branch_nodes,
                    bp_use_exact_pricing=config.branch_and_price.bp_use_exact_pricing,
                    bp_use_ng_routes=config.branch_and_price.bp_use_ng_routes,
                    bp_ng_neighborhood_size=config.branch_and_price.bp_ng_neighborhood_size,
                    bp_tree_search_strategy=config.branch_and_price.bp_tree_search_strategy,
                    bp_vehicle_limit=config.branch_and_price.bp_vehicle_limit,
                    bp_cleanup_frequency=config.branch_and_price.bp_cleanup_frequency,
                    bp_cleanup_threshold=config.branch_and_price.bp_cleanup_threshold,
                    bp_early_termination_gap=config.branch_and_price.bp_early_termination_gap,
                    bp_allow_heuristic_ryan_foster=config.branch_and_price.bp_allow_heuristic_ryan_foster,
                    bp_time_limit=config.branch_and_price.bp_time_limit,
                    bp_use_cspy=config.branch_and_price.bp_use_cspy,
                    seed=config.branch_and_price.seed,
                )
            elif method_lower in ["dp", "dp_route_reopt"]:
                processor = DPRouteReoptRouteImprover(
                    dp_max_nodes=config.dp_max_nodes,
                )
            elif method_lower == "fix_and_optimize":
                processor = FixAndOptimizeRouteImprover(
                    fo_n_free=config.fix_and_optimize.fo_n_free,
                    fo_free_fraction=config.fix_and_optimize.fo_free_fraction,
                    fo_time_limit=config.fix_and_optimize.fo_time_limit,
                    seed=config.fix_and_optimize.seed,
                )
            elif method_lower == "learned":
                processor = LearnedRouteImprover(
                    learned_weights_path=config.learned.learned_weights_path,
                    learned_max_iter=config.learned.learned_max_iter,
                    learned_min_improvement=config.learned.learned_min_improvement,
                    learned_neighborhood_size=config.learned.learned_neighborhood_size,
                    seed=config.learned.seed,
                )
            elif method_lower == "set_partitioning":
                processor = SetPartitioningRouteImprover(
                    sp_n_perturbations=config.set_partitioning.sp_n_perturbations,
                    sp_include_dp=config.set_partitioning.sp_include_dp,
                    sp_time_limit=config.set_partitioning.sp_time_limit,
                    ruin_fraction=config.set_partitioning.ruin_fraction,
                    seed=config.set_partitioning.seed,
                )
            elif method_lower == "set_partitioning_polish":
                processor = SetPartitioningPolishRouteImprover(
                    route_pool=config.set_partitioning_polish.route_pool,
                    sp_time_limit=config.set_partitioning_polish.sp_time_limit,
                    seed=config.set_partitioning_polish.seed,
                )
            elif method_lower == "node_exchange_steepest":
                processor = NodeExchangeSteepestRouteImprover()
            elif method_lower == "or_opt_steepest":
                processor = OrOptSteepestRouteImprover()
            elif method_lower == "steepest_two_opt":
                processor = SteepestTwoOptRouteImprover()
            elif method_lower == "mip_lns":
                processor = MIPLNSRouteImprover()
            elif method_lower == "neural_selector":
                processor = NeuralSelectorRouteImprover(config=config)
            elif method_lower == "adaptive_ensemble":
                processor = AdaptiveEnsembleRouteImprover()
            else:
                # Fallback to registry
                processor = cls.create(method)

            processors.append(processor)

        return processors

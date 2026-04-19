"""
Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) for MPVRP.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager
from logic.src.policies.helpers.operators.crossover_recombination.pattern_and_itinerary_crossover import (
    pattern_itinerary_crossover,
)
from logic.src.policies.helpers.operators.inter_route_local_search.inter_day_shift import inter_day_shift
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .individual import Individual
from .initialization import generate_initial_individual
from .population import Population
from .split import compute_daily_loads, split_day


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PARALLELIZABLE,
)
@RouteConstructorRegistry.register("hgs_adc")
class PolicyHGSADC(BaseMultiPeriodRoutingPolicy):
    """
    Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC).

    HGS-ADC is a state-of-the-art meta-heuristic for the Vehicle Routing Problem
    and its variants (Vidal et al., 2012). It combines the exploration power of
    Genetic Algorithms with the intensification of Local Search.

    Mathematical Principles:
    1.  **Unified Solution Representation**: Solutions are represented as giant
        tours (permutations of nodes) which are split into vehicle-feasible
        routes using an optimal Split procedure (Davis et al., 2003).
    2.  **Adaptive Diversity Control**: Instead of simple elitism, HGS-ADC
        maintains a population biased towards individuals that contribute
        the most to the population's structural diversity.
        Bias(B) = Rank(Cost(B)) + (1 - nb_elite / nb_total) * Rank(Diversity(B)).
    3.  **Feasibility Management**: The algorithm explores both feasible and
        infeasible solution spaces (penalizing capacity/duration violations)
        to bypass narrow corridors of feasibility.

    Registry key: ``"hgs_adc"``

    References:
        Vidal, T., et al. (2012). "A hybrid genetic algorithm with adaptive
        diversity management for a large class of vehicle routing problems with
        time windows". Computers & Operations Research, 39(9), 2125-2136.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC).

        HGS-ADC is a state-of-the-art metaheuristic for Vehicle Routing Problems.
        It combines the power of genetic algorithms (exploring the space of
        giant tours) with a "Split" algorithm (optimally partitioning tours
        into daily vehicle routes) and a diverse local search (education).

        Algorithmic Principles:
        1. **Individual Representation**: Encodes each solution as a sequence
           of daily giant tours (permutations of nodes).
        2. **Adaptive Diversity Control**: Manages two sub-populations (feasible
           and infeasible) based on objective value AND contribution to
           population diversity. This prevents premature convergence.
        3. **Split Algorithm**: Uses a dynamic programming approach to find the
           optimal vehicle routes for a given giant tour, ensuring maximum
           profit/minimum cost within capacity constraints.
        4. **Education (LS)**: Refines every child solution using local search
           operators (2-opt, SWAP*) to reach a local optimum.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - full_plan: Collection plan for the provided horizon.
                - stats: Execution statistics (generations, fitness).
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("HGS-ADC requires a ScenarioTree in ProblemContext.")

        capacity = problem.capacity
        dist_matrix = problem.distance_matrix

        T = tree.horizon
        base_wastes = tree.root.wastes.copy()
        N = len(base_wastes)

        daily_increments = np.zeros((T, N))
        previous_wastes = base_wastes

        for t in range(1, T + 1):
            scenarios = tree.get_scenarios_at_day(t)
            if scenarios:
                best_scenario = max(scenarios, key=lambda s: s.probability)
                current_wastes = best_scenario.wastes
                daily_increments[t - 1] = np.maximum(0, current_wastes - previous_wastes)
                previous_wastes = current_wastes

        best_ind = self._construct_multi_period_routes(T, base_wastes, daily_increments, dist_matrix, capacity)

        if best_ind is None:
            return SolutionContext.empty(), [[[0, 0]]] * T, {}

        full_plan = []
        for t in range(T):
            routes = best_ind.routes[t]
            day_plan: List[List[int]] = []
            for r in routes:
                if len(r) > 0:
                    day_plan.append([0] + list(r) + [0])
            if not day_plan:
                day_plan = [[0, 0]]
            full_plan.append(day_plan)

        today_route = full_plan[0][0] if full_plan[0] else []
        # Filter depot 0
        today_route = [v for v in today_route if v != 0]
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, full_plan, {"generations": getattr(self.config, "generations", 50)}

    def _construct_multi_period_routes(
        self,
        T: int,
        base_wastes: np.ndarray,
        daily_increments: np.ndarray,
        dist_matrix: np.ndarray,
        capacity: float,
        **kwargs: Any,
    ) -> Optional[Individual]:
        pop_size = getattr(self.config, "pop_size", 25)
        nb_close = getattr(self.config, "nb_close", 4)
        n_gen = getattr(self.config, "generations", 50)

        n_vehicles = 0
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            n_vehicles = kwargs["config"].get("n_vehicles", 0)
        if getattr(self.config, "n_vehicles", None) is not None:
            n_vehicles = self.config.n_vehicles

        N = len(base_wastes)
        population = Population(pop_size, nb_close)

        for _ in range(pop_size * 2):
            ind = generate_initial_individual(N, T, base_wastes, daily_increments, dist_matrix, capacity)
            self._evaluate_individual(ind, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)
            population.add_individual(ind)

        population.trigger_survivor_selection(T)

        for _gen in range(n_gen):
            pool = population.feas + population.inf
            if len(pool) < 2:
                break

            p1 = random.choice(pool)
            p2 = random.choice(pool)
            while p2 == p1 and len(pool) > 1:
                p2 = random.choice(pool)

            child = pattern_itinerary_crossover(p1, p2, T, N)

            self._evaluate_individual(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            loads = compute_daily_loads(child.patterns, base_wastes, daily_increments, T)
            improved_routes = self.run_intra_day_local_search(child.routes, dist_matrix, capacity, loads)

            child.routes = improved_routes
            for t in range(T):
                gt = []
                for r in improved_routes[t]:
                    gt.extend(r)
                child.giant_tours[t] = np.array(gt, dtype=int)

            self._evaluate_individual(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            if random.random() < 0.2:
                inter_day_shift(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            population.add_individual(child)

            if len(population.feas) + len(population.inf) > pop_size * 2:
                population.trigger_survivor_selection(T)

        if population.feas:
            return sorted(population.feas, key=lambda x: x.fit)[0]
        elif population.inf:
            return sorted(population.inf, key=lambda x: x.fit)[0]

        return None

    def _evaluate_individual(
        self,
        ind: Individual,
        base_wastes: np.ndarray,
        daily_increments: np.ndarray,
        dist: np.ndarray,
        capacity: float,
        n_vehicles: int,
        T: int,
    ) -> None:
        loads = compute_daily_loads(ind.patterns, base_wastes, daily_increments, T)
        tot_cost = 0.0
        tot_viol = 0.0
        all_routes = []

        for t in range(T):
            r, c, v = split_day(ind.giant_tours[t], loads[t], dist, capacity, n_vehicles)
            all_routes.append(r)
            tot_cost += c
            tot_viol += v

        w_q = 100.0
        ind.cost = tot_cost
        ind.capacity_violations = tot_viol
        ind.fit = tot_cost + w_q * tot_viol
        ind.is_feasible = tot_viol == 0.0
        ind.routes = all_routes

    @staticmethod
    def run_intra_day_local_search(
        routes: List[List[List[int]]],
        dist: np.ndarray,
        capacity: float,
        loads: np.ndarray,
    ) -> List[List[List[int]]]:
        """
        Intra-day Local Search (Education).
        Applies unified local search operators (2-opt, SWAP*)
        to the given decoded routes on a day-by-day basis.
        """
        improved_routes: List[List[List[int]]] = []
        T = len(routes)

        for t in range(T):
            t_routes = routes[t]
            if not t_routes:
                improved_routes.append([])
                continue

            load_array = loads[t]
            # Build waste dict for LocalSearchManager
            wastes = {i: float(load_array[i]) for i in range(len(load_array))}

            ls = LocalSearchManager(
                dist_matrix=dist, wastes=wastes, capacity=capacity, R=0.0, C=1.0, improvement_threshold=1e-4
            )
            ls.penalty_capacity = 100.0  # type: ignore
            ls.set_routes(t_routes)

            improved = True
            while improved:
                improved = False
                if ls.two_opt_intra():
                    improved = True
                if ls.swap_star():
                    improved = True

            improved_routes.append(ls.get_routes())

        return improved_routes

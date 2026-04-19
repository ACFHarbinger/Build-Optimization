"""
Reinforcement Learning Augmented Hybrid Volleyball Premier League (RL-AHVPL) with Contextual Multi-Armed Bandits for Crossover Selection.

This is the most advanced version of AHVPL, combining:
    1. Enhanced ACO with Q-Learning (initialization)
    2. Enhanced ALNS with SARSA (local search)
    3. **CMAB-HGS** (intelligent crossover operator selection)
    4. All previous improvements (profit-based pheromone, GLS, reactive tabu, etc.)

Crossover Operators (selected by CMAB):
    - OX (Ordered Crossover)
    - PIX (Position Independent Crossover)
    - SREX (Selective Route Exchange Crossover)
    - GPX (Generalized Partition Crossover)
    - ERX (Edge Recombination Crossover)

Bandit Algorithms:
    - LinUCB (Linear Upper Confidence Bound) - default
    - Thompson Sampling
    - ε-Greedy (baseline)

Reference:
    Implementation follows the deep analysis report recommendations.
    Li et al., "A Contextual-Bandit Approach", WWW 2010.
"""

import copy
import random
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.reinforcement_learning.alns_sarsa import ALNSSARSASolver
from logic.src.policies.helpers.reinforcement_learning.evolution_cmab import CMABEvolution, update_biased_fitness
from logic.src.policies.helpers.reinforcement_learning.ks_aco_qlearning import KSparseACOQLSolver
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit

from .params import RLAHVPLParams


class RLAHVPLSolver:
    """
    Reinforcement Learning Augmented Hybrid Volleyball Premier League.

    This version represents the culmination of all improvements:
    - Intelligent crossover selection via CMAB
    - Enhanced operators throughout the pipeline
    - Adaptive diversity management
    - Systematic local search
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RLAHVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize ultimate RL-AHVPL solver.

        Args:
            dist_matrix: Distance matrix.
            wastes: Node wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: RLAHVPLParams parameters.
            mandatory_nodes: Mandatory nodes to visit.
        """
        self.dist_matrix = np.array(dist_matrix)
        self.wastes = wastes.copy()
        if 0 not in self.wastes:
            self.wastes[0] = 0.0

        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Enhanced ACO with Q-Learning
        self.aco_solver = KSparseACOQLSolver(
            dist_matrix,
            self.wastes,
            capacity,
            R,
            C,
            params.aco_params,
            rl_params=params,
            mandatory_nodes=mandatory_nodes,
        )
        self.pheromone = self.aco_solver.pheromone
        self.constructor = self.aco_solver.constructor

        # Enhanced ALNS with SARSA
        self.alns_solver = ALNSSARSASolver(
            dist_matrix,
            self.wastes,
            capacity,
            R,
            C,
            params.alns_params,
            params,
            mandatory_nodes,
            evaluator=self._augmented_evaluate,
        )

        # HGS: LinearSplit for evaluation
        self.split_manager = LinearSplit(
            dist_matrix,
            self.wastes,
            capacity,
            R,
            C,
            params.hgs_params.max_vehicles,
            mandatory_nodes,
            params.vrpp,
        )

        cfe_params = {
            "alpha": self.params.cfe_alpha,
            "feature_dim": self.params.cfe_feature_dim,
            "operator_selection_threshold": self.params.cfe_operator_selection_threshold,
            "lambda_prior": self.params.cfe_lambda_prior,
            "noise_variance": self.params.cfe_noise_variance,
            "epsilon": self.params.cfe_epsilon,
            "epsilon_decay": self.params.cfe_epsilon_decay,
            "epsilon_min": self.params.cfe_epsilon_min,
            "diversity_history_size": self.params.cfe_diversity_history_size,
            "improvement_history_size": self.params.cfe_improvement_history_size,
            "operator_reward_size": self.params.cfe_operator_reward_size,
            "improvement_threshold": self.params.cfe_improvement_threshold,
        }

        # CMAB Evolution with intelligent crossover selection
        self.cmab_evolution = CMABEvolution(
            split_manager=self.split_manager,
            bandit_algorithm=self.params.bandit_algorithm,
            quality_weight=self.params.bandit_quality_weight,
            improvement_weight=self.params.bandit_improvement_weight,
            diversity_weight=self.params.bandit_diversity_weight,
            novelty_weight=self.params.bandit_novelty_weight,
            reward_threshold=self.params.bandit_reward_threshold,
            default_reward=self.params.bandit_default_reward,
            rng=self.random,
            **cfe_params,
        )

        # GLS edge penalties
        n = len(dist_matrix)
        self.edge_penalties = np.zeros((n, n), dtype=np.float64)

        # Reactive tabu memory
        self.solution_history: Dict[int, int] = {}
        self.tabu_list: Deque[int] = deque(maxlen=self.params.rts_params.max_tenure)

        # Tracking
        self.best_profit_history: List[float] = []

    # ===== Initialization =====

    def _initialize_population(self) -> List[Individual]:
        """Generate initial population using enhanced ACO."""
        population: List[Individual] = []
        # Target size is n_teams
        for _ in range(self.params.n_teams):
            ind = self._construct_individual()
            if ind is not None:
                population.append(ind)

        # Ensure minimum population size for evolution
        if len(population) < 2:
            # Fallback: add some simple randomized individuals if ACO fails to provide enough
            while len(population) < min(2, self.params.n_teams):
                shuffled = self.nodes[:]
                self.random.shuffle(shuffled)
                population.append(Individual(shuffled))

        return population

    def _construct_individual(self) -> Optional[Individual]:
        """Build Individual from ACO-Q solution."""
        routes, profit, cost = self.aco_solver.solve()
        if not routes:
            return None

        giant_tour = self._routes_to_giant_tour(routes)
        if not giant_tour:
            return None

        # Ensure all nodes present
        visited = set(giant_tour)
        missing = [n for n in self.nodes if n not in visited]
        giant_tour.extend(missing)

        ind = Individual(giant_tour)
        ind.routes = [r[:] for r in routes]
        ind.profit_score = profit
        ind.cost = cost
        ind.revenue = profit + cost

        return ind

    # ===== Main Solve Loop =====

    def solve(self) -> Tuple[List[List[int]], float, float]:  # noqa: C901
        """
        Run Ultimate AHVPL with CMAB crossover selection.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        start_time = time.process_time()

        # Phase 1: Enhanced ACO initialization
        population = self._initialize_population()

        if not population:
            return [], 0.0, 0.0

        best_ind = max(population, key=lambda x: x.profit_score)
        best_routes = [r[:] for r in best_ind.routes]
        best_profit = best_ind.profit_score
        best_cost = best_ind.cost
        self.best_profit_history.append(best_profit)

        no_repeat_count = 0
        tenure = self.params.rts_params.initial_tenure
        iteration = 0
        it_no_improvement = 0
        while it_no_improvement < self.params.hgs_params.n_iterations_no_improvement:
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break
            iteration += 1
            it_no_improvement += 1

            prev_best_profit = best_profit

            # 1. Bi-criteria fitness update
            update_biased_fitness(
                population,
                self.params.hgs_params.nb_elite,
                self.params.hgs_params.nb_granular,
            )

            # 2. CMAB-driven crossover
            n_crossovers = max(1, int(len(population) * self.params.hgs_params.crossover_rate))
            n_children = 0
            for _ in range(n_crossovers):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                # Select parents
                p1, p2 = self._select_parents(population)

                # Calculate progress based on time
                progress = (
                    (time.process_time() - start_time) / self.params.time_limit if self.params.time_limit > 0 else 0.0
                )

                # CMAB selects crossover operator and creates child
                child = self.cmab_evolution.crossover(p1, p2, population, iteration, progress)

                # Mutation
                if self.random.random() < self.params.hgs_params.mutation_rate:
                    self._mutate(child)
                    self.cmab_evolution.evaluate(child)

                population.append(child)
                n_children += 1

                # Reactive tabu - cycle detection
                child_hash = self._hash_solution(child)
                is_tabu = child_hash in self.tabu_list

                # Aspiration criterion: if it's tabu and doesn't improve global best, force mutation
                if is_tabu and child.profit_score <= best_profit:
                    self._mutate(child)
                    self._mutate(child)  # Heavy mutation to escape
                    self.cmab_evolution.evaluate(child)
                    child_hash = self._hash_solution(child)

                if child_hash in self.solution_history:
                    # Cycle detected - increase tenure
                    tenure = min(
                        self.params.rts_params.max_tenure, int(tenure * self.params.rts_params.tenure_increase)
                    )
                    no_repeat_count = 0
                else:
                    no_repeat_count += 1
                    if no_repeat_count > self.params.tabu_no_repeat_threshold * tenure:
                        # Long non-cycling - decrease tenure
                        tenure = max(
                            self.params.rts_params.min_tenure, int(tenure * self.params.rts_params.tenure_decrease)
                        )
                        no_repeat_count = 0
                    self.solution_history[child_hash] = iteration

                self.tabu_list.append(child_hash)
                # Trim tabu list to current dynamic tenure
                while len(self.tabu_list) > tenure:
                    self.tabu_list.popleft()

            # 3. Adaptive coaching (VND for elites, ALNS-SARSA for others)
            population.sort(key=lambda x: (x.profit_score, tuple(tuple(r) for r in x.routes)), reverse=True)
            for i, ind in enumerate(population):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                if i < self.params.hgs_params.nb_elite:
                    # Elite -> Intensive ALNS
                    new_ind = self._alns_coaching(ind, iterations=self.params.elite_coaching_max_iterations)
                elif self.random.random() < self.params.gls_probability:
                    # Others -> Quick GLS
                    new_ind = self._gls_coaching(ind, iterations=self.params.not_coached_max_iterations)
                else:
                    # Already coached: light refresh
                    new_ind = self._alns_coaching(ind, iterations=self.params.alns_params.max_iterations)
                population[i] = new_ind

            # 4. Survivor selection
            update_biased_fitness(
                population,
                self.params.n_teams,
                self.params.hgs_params.nb_granular,
            )
            population.sort(key=lambda x: x.fitness)
            population = population[: self.params.n_teams]

            # 5. Reduced substitution
            population.sort(key=lambda x: x.profit_score, reverse=True)
            n_sub = int(self.params.n_teams * self.params.sub_rate)
            for i in range(self.params.n_teams - n_sub, self.params.n_teams):
                new_ind = self._construct_individual()  # type: ignore[assignment]
                if new_ind:
                    population[i] = new_ind

            # 6. Update global best
            iter_best = max(population, key=lambda x: x.profit_score)
            if iter_best.profit_score > best_profit:
                best_routes = [r[:] for r in iter_best.routes]
                best_profit = iter_best.profit_score
                best_cost = iter_best.cost
                it_no_improvement = 0

            self.best_profit_history.append(best_profit)

            # 7. Profit-based pheromone update
            self._update_pheromones_profit(best_routes, best_profit, best_cost)

            # 8. GLS: penalize if stagnating
            if iteration > 0 and iteration % self.params.gls_penalty_step == 0 and best_profit == prev_best_profit:
                self._penalize_local_optimum_edges(best_routes)

            # 9. Update CMAB improvement tracking
            improvement_rate = (best_profit - prev_best_profit) / max(
                abs(prev_best_profit), self.params.cfe_improvement_threshold
            )
            self.cmab_evolution.update_improvement(improvement_rate)

            # 10. Decay exploration
            if iteration % self.params.cfe_epsilon_decay_step == 0:
                self.cmab_evolution.decay_exploration()

            # Visualization with CMAB stats
            cmab_stats = self.cmab_evolution.get_statistics()
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iter_best.profit_score,
                population_size=len(population),
                n_children=n_children,
                tabu_tenure=tenure,
                bandit_algorithm=self.params.bandit_algorithm,
                cmab_stats=cmab_stats,
            )

        return best_routes, best_profit, best_cost

    # ===== Coaching Methods =====

    def _gls_coaching(self, ind: Individual, iterations: int = 100) -> Individual:
        """Guided Local Search coaching using augmented objective."""
        routes = [r[:] for r in ind.routes]
        best_routes = [r[:] for r in routes]

        profit = ind.profit_score
        best_profit = profit
        aug_profit = self._augmented_evaluate(routes)

        # Simplified GLS operators pool using existing ALNS operators
        llhs = [
            (self.alns_solver.destroy_ops[0], self.alns_solver.repair_ops[0]),  # random + greedy
            (self.alns_solver.destroy_ops[1], self.alns_solver.repair_ops[1]),  # worst + regret2
            (self.alns_solver.destroy_ops[2], self.alns_solver.repair_ops[0]),  # cluster + greedy
            (self.alns_solver.destroy_ops[1], self.alns_solver.repair_ops[0]),  # worst + greedy
            (self.alns_solver.destroy_ops[0], self.alns_solver.repair_ops[1]),  # random + regret2
        ]

        n_remove = max(1, int(self.n_nodes * self.params.alns_params.max_removal_pct))

        for _ in range(iterations):
            d_op, r_op = self.random.choice(llhs)

            try:
                partial_routes, removed = d_op(copy.deepcopy(routes), n_remove)
                new_routes = r_op(partial_routes, removed)
            except Exception:
                continue

            new_aug_profit = self._augmented_evaluate(new_routes)

            if new_aug_profit >= aug_profit:
                routes = new_routes
                aug_profit = new_aug_profit

                new_real_profit = self._evaluate_routes(routes)
                if new_real_profit > best_profit:
                    best_routes = [r[:] for r in routes]
                    best_profit = new_real_profit

        # At local optimum, penalize features
        self._penalize_local_optimum_edges(routes)

        if best_profit > ind.profit_score:
            ind.routes = best_routes
            ind.profit_score = best_profit
            ind.cost = self._calculate_cost(best_routes)
            ind.revenue = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)

            ind.giant_tour = self._routes_to_giant_tour(best_routes)
            visited = set(ind.giant_tour)
            missing = [n for n in self.nodes if n not in visited]
            ind.giant_tour.extend(missing)

        ind.is_coached = True
        return ind

    def _alns_coaching(self, ind: Individual, iterations: int = 100) -> Individual:
        """Enhanced ALNS-SARSA coaching."""
        old_iters = self.alns_solver.params.max_iterations
        self.alns_solver.params.max_iterations = iterations

        improved_routes, improved_profit, improved_cost = self.alns_solver.solve(initial_solution=ind.routes)

        self.alns_solver.params.max_iterations = old_iters

        if improved_profit > ind.profit_score + self.params.coaching_acceptance_threshold and improved_routes:
            ind.routes = [r[:] for r in improved_routes]
            ind.profit_score = improved_profit
            ind.cost = improved_cost
            ind.revenue = improved_profit + improved_cost

            ind.giant_tour = self._routes_to_giant_tour(improved_routes)
            visited = set(ind.giant_tour)
            missing = [n for n in self.nodes if n not in visited]
            ind.giant_tour.extend(missing)

        ind.is_coached = True
        return ind

    # ===== Helper Methods =====

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Binary tournament selection."""
        if len(population) < 2:
            # This should not happen with current safeguards, but handle for safety
            p = population[0] if population else Individual([])
            return p, p

        def tournament() -> Individual:
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    def _mutate(self, ind: Individual) -> None:
        """SWAP mutation."""
        size = len(ind.giant_tour)
        if size < 2:
            return
        idx1, idx2 = self.random.sample(range(size), 2)
        ind.giant_tour[idx1], ind.giant_tour[idx2] = ind.giant_tour[idx2], ind.giant_tour[idx1]
        ind.is_coached = False

    def _hash_solution(self, ind: Individual) -> int:
        """Hash for cycle detection."""
        return hash(tuple(tuple(r) for r in ind.routes))

    def _penalize_local_optimum_edges(self, routes: List[List[int]]) -> None:
        """GLS penalty update."""
        edges = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))

        if not edges:
            return

        best_utility = -1.0
        best_edge = None
        for i, j in edges:
            cost_ij = self.dist_matrix[i][j]
            utility = cost_ij / (1.0 + self.edge_penalties[i][j])
            if utility > best_utility:
                best_utility = utility
                best_edge = (i, j)

        if best_edge is not None:
            self.edge_penalties[best_edge[0]][best_edge[1]] += 1.0

    def _augmented_evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate with penalty-augmented objective."""
        real = self._evaluate_routes(routes)
        penalty = 0.0
        dynamic_lambda = self.params.gls_penalty_lambda * (abs(real) / max(1, self.n_nodes))
        for route in routes:
            if not route:
                continue
            penalty += self.edge_penalties[0][route[0]]
            for k in range(len(route) - 1):
                penalty += self.edge_penalties[route[k]][route[k + 1]]
            penalty += self.edge_penalties[route[-1]][0]
        return real - self.params.gls_penalty_lambda * dynamic_lambda * penalty

    def _update_pheromones_profit(self, routes: List[List[int]], profit: float, cost: float) -> None:
        """Profit-based pheromone update."""
        if not routes or profit <= 0:
            return

        self.pheromone.evaporate_all(self.params.aco_params.rho)
        delta = self.params.aco_params.elitist_weight * profit / (cost + 1e-6)

        for route in routes:
            if not route:
                continue
            self.pheromone.deposit_edge(0, route[0], delta)
            for k in range(len(route) - 1):
                self.pheromone.deposit_edge(route[k], route[k + 1], delta)
            self.pheromone.deposit_edge(route[-1], 0, delta)

    @staticmethod
    def _routes_to_giant_tour(routes: List[List[int]]) -> List[int]:
        """Flatten routes, ensuring depot (0) is filtered and duplicates removed."""
        gt: List[int] = []
        seen = {0}
        for route in routes:
            for node in route:
                if node not in seen:
                    gt.append(node)
                    seen.add(node)
        return gt

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total * self.C

    def _evaluate_routes(self, routes: List[List[int]]) -> float:
        """Calculate profit."""
        cost = self._calculate_cost(routes)
        revenue = sum(self.wastes.get(n, 0) * self.R for r in routes for n in r)
        return revenue - cost

"""
Genetic Programming Hyper-Heuristic (GPHH) for Build Optimization.

Instead of evolving solutions, GPHH evolves selection *policies* — GP
expression trees that decide which Low-Level Heuristic (LLH) to apply.
"""

import random
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import GPHHParams
from .tree import GPNode, _mutate, _random_tree, _subtree_crossover


class GPHHSolver(PolicyVizMixin):
    """
    Genetic Programming Hyper-Heuristic solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: GPHHParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # LLH pool
        self._llh_pool: List[Callable] = [
            self._llh_greedy,
            self._llh_regret,
            self._llh_blink,
        ]
        # Actual pool size
        self.n_llh = len(self._llh_pool)

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run GPHH: evolve LLH selection policies, then apply best.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        init_build = self.problem.greedy_solution()

        # Initialise GP population
        gp_pop = [_random_tree(self.params.tree_depth, self.n_llh) for _ in range(self.params.gp_pop_size)]
        gp_fitness = [self._evaluate_tree(tree, init_build, self.params.eval_steps) for tree in gp_pop]

        best_tree_idx = int(np.argmax(gp_fitness))
        best_tree = gp_pop[best_tree_idx].copy()
        best_tree_fitness = gp_fitness[best_tree_idx]

        # GP evolution loop
        for gen in range(self.params.max_gp_generations):
            if time.time() - start > self.params.time_limit * 0.6:
                break

            new_pop: List[GPNode] = []
            new_fitness: List[float] = []

            while len(new_pop) < self.params.gp_pop_size:
                # Tournament selection
                p1 = self._tournament(gp_pop, gp_fitness)
                p2 = self._tournament(gp_pop, gp_fitness)

                # Crossover
                c1, c2 = _subtree_crossover(p1.copy(), p2.copy())

                # Mutation
                if random.random() < 0.3:
                    c1 = _mutate(c1, self.params.tree_depth, self.n_llh)
                if random.random() < 0.3:
                    c2 = _mutate(c2, self.params.tree_depth, self.n_llh)

                f1 = self._evaluate_tree(c1, init_build, self.params.eval_steps)
                f2 = self._evaluate_tree(c2, init_build, self.params.eval_steps)

                new_pop.extend([c1, c2])
                new_fitness.extend([f1, f2])

                for tree, fitness in [(c1, f1), (c2, f2)]:
                    if fitness > best_tree_fitness:
                        best_tree = tree.copy()
                        best_tree_fitness = fitness

            gp_pop = new_pop[: self.params.gp_pop_size]
            gp_fitness = new_fitness[: self.params.gp_pop_size]

            self._viz_record(
                iteration=gen,
                best_tree_fitness=best_tree_fitness,
                gp_pop_size=self.params.gp_pop_size,
            )

        # Apply best tree for the final run
        best_build, best_score = self._apply_tree(
            best_tree,
            init_build,
            self.params.apply_steps,
        )

        return best_build, best_score

    def _evaluate_tree(self, tree: GPNode, init_build: np.ndarray, n_steps: int) -> float:
        """Evaluate a tree by achievement in n_steps."""
        _, best_score = self._apply_tree(tree, init_build.copy(), n_steps)
        return best_score

    def _apply_tree(
        self,
        tree: GPNode,
        init_build: np.ndarray,
        n_steps: int,
    ) -> Tuple[np.ndarray, float]:
        """Apply policy for n_steps LLH calls."""
        build = init_build.copy()
        score = self.problem.evaluate(build)
        best_build = build.copy()
        best_score = score

        for step in range(n_steps):
            ctx = self._build_context(build, step, n_steps)
            llh_idx = int(round(tree.evaluate(ctx))) % self.n_llh
            llh = self._llh_pool[llh_idx]

            try:
                new_build = llh(build.copy())
                new_score = self.problem.evaluate(new_build)
                # Accept improvement (greedy)
                if new_score >= score:
                    build = new_build
                    score = new_score
                    if score > best_score:
                        best_build = build.copy()
                        best_score = score
            except Exception:
                pass

        return best_build, best_score

    def _build_context(self, build: np.ndarray, step: int, total: int) -> Dict[str, float]:
        """Build feature context for the GP tree."""
        score = self.problem.evaluate(build)
        cost = self.problem.budget_used(build)
        num_filled = np.sum(build != -1)

        return {
            "avg_item_score": score / max(num_filled, 1),
            "budget_ratio": cost / self.budget,
            "fill_ratio": num_filled / self.problem.num_slots,
            "iter_progress": float(step) / max(float(total), 1.0),
        }

    def _tournament(self, pop: List[GPNode], fitness: List[float]) -> GPNode:
        k = min(self.params.tournament_size, len(pop))
        candidates = random.sample(range(len(pop)), k)
        best = max(candidates, key=lambda i: fitness[i])
        return pop[best]

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)

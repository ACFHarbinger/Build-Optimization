"""
Genetic Programming Hyper-Heuristic (GPHH) for VRPP.

Instead of evolving solutions, GPHH evolves selection *policies* — GP
expression trees that decide which Low-Level Heuristic (LLH) to apply
at each step based on observable routing features.

Hyper-heuristic elevates the abstraction: the GP tree controls a set of
LLHs (destroy + repair operators) rather than modifying routes directly.
This produces generalised policies that adapt to real-time state.

LLH Pool:
  L0: random_removal  + greedy_insertion
  L1: worst_removal   + regret_2_insertion
  L2: cluster_removal + greedy_insertion
  L3: worst_removal   + greedy_insertion
  L4: random_removal  + regret_2_insertion

GP Tree Nodes:
  Terminals: avg_node_profit, load_factor, route_count, iter_progress
  Functions: IF_GT(a, b, L_i, L_j) — selects LLH L_i if a>b else L_j
             MAX_LLH(a, b) — returns argmax(a, b) as LLH index

Reference:
    Survey §"Genetic Programming Hyper-Heuristics" — stochastic TOP w/ time windows.
"""

import copy
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .params import GPHHParams
from .tree import GPNode, _mutate, _random_tree, _subtree_crossover

# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


class GPHHSolver(PolicyVizMixin):
    """
    Genetic Programming Hyper-Heuristic solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GPHHParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # LLH pool (each is a callable: routes, n_remove → routes)
        self._llh_pool: List[Callable] = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run GPHH: evolve LLH selection policies, then apply best.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.time()

        # Initial solution
        init_routes = self._random_solution()

        # Initialise GP population
        gp_pop = [_random_tree(self.params.tree_depth, self.params.n_llh) for _ in range(self.params.gp_pop_size)]
        gp_fitness = [self._evaluate_tree(tree, init_routes, self.params.eval_steps) for tree in gp_pop]

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
                    c1 = _mutate(c1, self.params.tree_depth, self.params.n_llh)
                if random.random() < 0.3:
                    c2 = _mutate(c2, self.params.tree_depth, self.params.n_llh)

                f1 = self._evaluate_tree(c1, init_routes, self.params.eval_steps)
                f2 = self._evaluate_tree(c2, init_routes, self.params.eval_steps)

                new_pop.extend([c1, c2])
                new_fitness.extend([f1, f2])

                # Update best tree
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
        best_routes, best_profit = self._apply_tree(
            best_tree,
            init_routes,
            self.params.apply_steps,
        )

        best_profit = self._evaluate(best_routes)
        best_cost = self._cost(best_routes)

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Tree evaluation and application
    # ------------------------------------------------------------------

    def _evaluate_tree(self, tree: GPNode, init_routes: List[List[int]], n_steps: int) -> float:
        """
        Evaluate a GP tree by running its LLH selection policy for n_steps.

        Args:
            tree: GP selection policy tree.
            init_routes: Starting routing solution.
            n_steps: Number of LLH applications.

        Returns:
            Best profit achieved during the run.
        """
        routes, best_profit = self._apply_tree(tree, copy.deepcopy(init_routes), n_steps)
        return best_profit

    def _apply_tree(
        self,
        tree: GPNode,
        init_routes: List[List[int]],
        n_steps: int,
    ) -> Tuple[List[List[int]], float]:
        """
        Apply a GP selection policy for n_steps LLH calls.

        Args:
            tree: GP selection policy tree.
            init_routes: Starting routing solution.
            n_steps: LLH application count.

        Returns:
            Tuple of (best_routes, best_profit).
        """
        routes = copy.deepcopy(init_routes)
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        for step in range(n_steps):
            ctx = self._build_context(routes, step, n_steps)
            llh_idx = int(round(tree.evaluate(ctx))) % self.params.n_llh
            llh = self._llh_pool[llh_idx]

            try:
                new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                new_profit = self._evaluate(new_routes)
                # Accept improvement (greedy acceptance)
                if new_profit >= profit:
                    routes = new_routes
                    profit = new_profit
                    if profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = profit
            except Exception:
                pass

        return best_routes, best_profit

    def _build_context(self, routes: List[List[int]], step: int, total: int) -> Dict[str, float]:
        """
        Build feature context for the GP tree.

        Args:
            routes: Current routing solution.
            step: Current iteration index.
            total: Total iterations.

        Returns:
            Feature dict.
        """
        all_nodes = [n for r in routes for n in r]
        n = max(len(all_nodes), 1)
        avg_profit = sum(self.wastes.get(nd, 0.0) * self.R for nd in all_nodes) / n
        total_load = sum(self.wastes.get(nd, 0.0) for nd in all_nodes)
        return {
            "avg_node_profit": avg_profit,
            "load_factor": total_load / max(self.capacity, 1e-9),
            "route_count": float(len(routes)),
            "iter_progress": float(step) / max(float(total), 1.0),
        }

    def _tournament(self, pop: List[GPNode], fitness: List[float]) -> GPNode:
        """Tournament selection from the GP population."""
        k = min(self.params.tournament_size, len(pop))
        candidates = random.sample(range(len(pop)), k)
        best = max(candidates, key=lambda i: fitness[i])
        return pop[best]

    # ------------------------------------------------------------------
    # LLH definitions
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L0: random_removal + greedy_insertion."""
        partial, removed = random_removal(routes, n)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L1: worst_removal + regret_2_insertion."""
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L2: cluster_removal + greedy_insertion."""
        partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L3: worst_removal + greedy_insertion."""
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """L4: random_removal + regret_2_insertion."""
        partial, removed = random_removal(routes, n)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_solution(self) -> List[List[int]]:
        """Generate a random feasible routing solution."""
        return self._build_random_solution()

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        from logic.src.policies.operators.heuristics.initialization import build_nn_routes

        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return optimized_routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total

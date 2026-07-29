"""
SolvingState: executes the optimization algorithm on the BuildProblem.

Dispatches to the appropriate build solver based on ``ctx.solver_name``
and stores the best solution found on ``ctx.best_build`` / ``ctx.best_score``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.problem import BuildProblem

    from ..context import GameOptimizationContext

from .base import GameState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

_SOLVERS: Dict[str, Any] = {}


def _register(name: str):
    def decorator(fn):
        _SOLVERS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Greedy solver
# ---------------------------------------------------------------------------


@_register("greedy")
def _solve_greedy(
    problem: "BuildProblem",
    config: Dict[str, Any],
    time_limit: float,
) -> Tuple[np.ndarray, float]:
    """Deterministic greedy: pick best affordable item per slot in order."""
    solution = problem.greedy_solution()
    score = problem.score_fast(solution)
    return solution, score


# ---------------------------------------------------------------------------
# Exact solver (C++ backend, branch-and-bound MCKP)
# ---------------------------------------------------------------------------


@_register("bnb")
def _solve_bnb(
    problem: "BuildProblem",
    config: Dict[str, Any],
    time_limit: float,
) -> Tuple[np.ndarray, float]:
    """Exact solve via the C++ branch-and-bound MCKP solver (backend/mckp.cpp).

    Each equipment slot is an MCKP class and each item a class option, which
    is an exact model of `Build`'s one-item-per-slot structure. Optimizes
    `score_fast` (the additive per-item objective every solver here uses
    internally) to global optimality; synergy bonuses are layered on
    afterward via `score_full`, the same as every other solver.

    Requires the backend extension to be built (see
    `core.native_backend.load_backend`); raises `ImportError` with build
    instructions if it isn't.
    """
    from core.native_backend import load_backend

    backend = load_backend()

    options = []
    for item_idx in range(problem.num_items):
        value = float(np.dot(problem.stat_matrix[item_idx], problem.stat_weights))
        value += problem.slot_bonus
        value += float(problem.rarities[item_idx]) * problem.rarity_bonus
        weight = int(round(float(problem.costs[item_idx])))
        options.append(backend.MckpOption(int(problem.slot_ids[item_idx]), weight, value))

    capacity = int(round(problem.budget)) if math.isfinite(problem.budget) else int(problem.costs.sum()) + 1

    result = backend.solve_mckp_branch_and_bound(options, problem.num_slots, capacity)

    solution = np.full(problem.num_slots, -1, dtype=np.int64)
    # `selected_option_indices` refers to indices into `options`, which was
    # built 1:1 aligned with item indices — no separate mapping needed.
    for item_idx in result.selected_option_indices:
        solution[int(problem.slot_ids[item_idx])] = item_idx

    score = problem.score_fast(solution)
    return solution, score


# ---------------------------------------------------------------------------
# Simulated Annealing solver
# ---------------------------------------------------------------------------


@_register("sa")
def _solve_sa(
    problem: "BuildProblem",
    config: Dict[str, Any],
    time_limit: float,
) -> Tuple[np.ndarray, float]:
    """Simulated Annealing over slot-to-item assignments."""
    rng = np.random.default_rng(config.get("seed"))

    T = float(config.get("initial_temp", 2.0))
    alpha = float(config.get("alpha", 0.995))
    T_min = float(config.get("min_temp", 1e-4))
    max_iter = int(config.get("max_iterations", 10_000))

    current = problem.greedy_solution()
    current_score = problem.score_fast(current)
    best = current.copy()
    best_score = current_score

    deadline = time.time() + time_limit

    for _ in range(max_iter):
        if time.time() >= deadline or T_min >= T:
            break

        neighbor = _random_move(current, problem, rng)
        if not problem.is_feasible(neighbor):
            T *= alpha
            continue

        neighbor_score = problem.score_fast(neighbor)
        delta = neighbor_score - current_score

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        T *= alpha

    logger.debug("SA finished: score=%.4f", best_score)
    return best, best_score


# ---------------------------------------------------------------------------
# Genetic Algorithm solver
# ---------------------------------------------------------------------------


@_register("ga")
def _solve_ga(
    problem: "BuildProblem",
    config: Dict[str, Any],
    time_limit: float,
) -> Tuple[np.ndarray, float]:
    """Genetic Algorithm over slot-to-item assignments."""
    rng = np.random.default_rng(config.get("seed"))

    pop_size = int(config.get("pop_size", 50))
    max_gen = int(config.get("max_generations", 200))
    cx_rate = float(config.get("crossover_rate", 0.8))
    mut_rate = float(config.get("mutation_rate", 0.2))
    tourn_size = int(config.get("tournament_size", 3))
    elite_n = max(1, int(pop_size * float(config.get("elite_frac", 0.1))))

    # Initialise population: one greedy seed + randoms
    population: List[np.ndarray] = [problem.greedy_solution()]
    while len(population) < pop_size:
        population.append(problem.random_solution(rng))

    scores = [problem.score_fast(ind) for ind in population]
    best_idx = int(np.argmax(scores))
    best = population[best_idx].copy()
    best_score = scores[best_idx]

    deadline = time.time() + time_limit

    for _ in range(max_gen):
        if time.time() >= deadline:
            break

        # Elite preservation
        elite_indices = np.argsort(scores)[-elite_n:]
        next_pop: List[np.ndarray] = [population[i].copy() for i in elite_indices]

        while len(next_pop) < pop_size:
            p1 = _tournament(population, scores, tourn_size, rng)
            p2 = _tournament(population, scores, tourn_size, rng)

            child = _uniform_crossover(p1, p2, rng) if rng.random() < cx_rate else p1.copy()

            if rng.random() < mut_rate:
                child = _random_move(child, problem, rng)

            child = _repair_budget(child, problem)
            next_pop.append(child)

        population = next_pop
        scores = [problem.score_fast(ind) for ind in population]
        gen_best = int(np.argmax(scores))
        if scores[gen_best] > best_score:
            best = population[gen_best].copy()
            best_score = scores[gen_best]

    logger.debug("GA finished: score=%.4f", best_score)
    return best, best_score


# ---------------------------------------------------------------------------
# Move operators (shared by SA and GA)
# ---------------------------------------------------------------------------


def _random_move(
    solution: np.ndarray,
    problem: "BuildProblem",
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one of three random moves: swap, fill, or drop."""
    new_sol = solution.copy()
    move = rng.integers(0, 3)

    if move == 0:
        # Swap: replace item in a random filled slot
        filled = np.where(new_sol >= 0)[0]
        if len(filled) == 0:
            return new_sol
        slot_idx = int(rng.choice(filled))
        candidates = np.where(problem.slot_ids == slot_idx)[0]
        if len(candidates) > 0:
            new_sol[slot_idx] = int(rng.choice(candidates))

    elif move == 1:
        # Fill: assign an item to a random empty slot
        empty = np.where(new_sol < 0)[0]
        if len(empty) == 0:
            return new_sol
        slot_idx = int(rng.choice(empty))
        candidates = np.where(problem.slot_ids == slot_idx)[0]
        if len(candidates) > 0:
            new_sol[slot_idx] = int(rng.choice(candidates))

    else:
        # Drop: unequip a random filled slot (keep at least one item)
        filled = np.where(new_sol >= 0)[0]
        if len(filled) > 1:
            new_sol[int(rng.choice(filled))] = -1

    return new_sol


def _tournament(
    population: List[np.ndarray],
    scores: List[float],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tournament selection: pick the best individual from ``size`` random candidates."""
    indices = rng.integers(0, len(population), size=size)
    winner = indices[int(np.argmax([scores[i] for i in indices]))]
    return population[int(winner)]


def _uniform_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform slot crossover: each slot is independently inherited from one parent."""
    mask = rng.integers(0, 2, size=p1.shape, dtype=bool)
    child = np.where(mask, p1, p2)
    return child


def _repair_budget(solution: np.ndarray, problem: "BuildProblem") -> np.ndarray:
    """Drop the most expensive items until the solution is budget-feasible."""
    sol = solution.copy()
    while not problem.is_feasible(sol):
        filled = np.where(sol >= 0)[0]
        if len(filled) == 0:
            break
        # Drop the most expensive equipped item
        costs = problem.costs[sol[filled]]
        drop_slot = filled[int(np.argmax(costs))]
        sol[int(drop_slot)] = -1
    return sol


# ---------------------------------------------------------------------------
# SolvingState
# ---------------------------------------------------------------------------


class SolvingState(GameState):
    """Runs the chosen optimization algorithm on the BuildProblem."""

    def handle(self, ctx: "GameOptimizationContext") -> None:
        if ctx.problem is None:
            logger.error("No BuildProblem available — skipping solve.")
            ctx.result = {"success": False, "error": "No problem instance"}
            ctx.transition_to(None)
            return

        solver_name = ctx.solver_name.lower()
        solver_fn = _SOLVERS.get(solver_name)

        if solver_fn is None:
            logger.error("Unknown solver %r. Available: %s", solver_name, list(_SOLVERS))
            ctx.result = {"success": False, "error": f"Unknown solver: {solver_name!r}"}
            ctx.transition_to(None)
            return

        logger.info("Running solver: %s (time_limit=%.1fs)", solver_name, ctx.time_limit)
        start = time.time()
        slot_to_item, fast_score = solver_fn(ctx.problem, ctx.solver_config, ctx.time_limit)
        ctx.elapsed = time.time() - start

        # Convert to Build and compute full score (with synergy bonuses)
        ctx.best_build = ctx.problem.to_build(slot_to_item, ctx.character_level)
        ctx.best_score = ctx.problem.score_full(slot_to_item, ctx.character_level)
        ctx.extra = {
            "fast_score": fast_score,
            "slot_to_item": slot_to_item.tolist(),
        }

        logger.info(
            "Solver %s finished in %.2fs — score=%.4f, cost=%.0f",
            solver_name,
            ctx.elapsed,
            ctx.best_score,
            ctx.best_build.total_cost if ctx.best_build else 0.0,
        )

        from .reporting import ReportingState

        ctx.transition_to(ReportingState())

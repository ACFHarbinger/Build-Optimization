"""
Game Build Optimization — top-level entry points.

Analogous to ``simulator.py`` in the simulation pipeline, this module
provides clean functions for running the full optimization pipeline
(Loading → Solving → Reporting) without manually constructing the
state machine.

Usage::

    from pipeline.games import run_optimization, run_batch

    # Single solver
    result = run_optimization(
        solver_name="sa",
        items_path="src/data/sample/rpg.json",
        budget=1500.0,
        character_level=50,
        verbose=True,
    )
    print(f"Score: {result['score']:.2f}")
    print(f"Cost:  {result['cost']:.0f} / {result['budget']:.0f}")

    # Compare multiple solvers on the same instance
    results = run_batch(
        solver_names=["greedy", "sa", "ga"],
        items_path="src/data/sample/rpg.json",
        budget=2000.0,
        verbose=True,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.scoring import ScoringConfig

from .context import GameOptimizationContext
from .states.loading import LoadingState

logger = logging.getLogger(__name__)


def run_optimization(
    solver_name: str,
    items_path: str,
    budget: float = float("inf"),
    character_level: int = 99,
    time_limit: float = 30.0,
    scoring_config: Optional[ScoringConfig] = None,
    solver_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one build optimization pass and return the best build found.

    Creates a GameOptimizationContext, runs the three-state machine
    (LoadingState → SolvingState → ReportingState), and returns the
    final result dictionary.

    Args:
        solver_name:    Algorithm to use: ``"greedy"``, ``"sa"``, or ``"ga"``.
        items_path:     Path to the game data file (JSON or CSV).
        budget:         Maximum total item cost.
        character_level: Items above this level are excluded.
        time_limit:     Solver wall-clock limit in seconds.
        scoring_config: Stat weights and bonuses; defaults to ScoringConfig().
        solver_config:  Algorithm hyperparameters (merged with defaults).
        verbose:        If True, print a formatted result summary.

    Returns:
        Result dict with keys: ``success``, ``solver``, ``score``, ``cost``,
        ``budget``, ``items_equipped``, ``elapsed``, ``active_synergies``,
        ``build``, ``extra``, ``slots``, ``total_stats``.
    """
    ctx = GameOptimizationContext(
        solver_name=solver_name,
        items_path=items_path,
        budget=budget,
        character_level=character_level,
        time_limit=time_limit,
        scoring_config=scoring_config or ScoringConfig(),
        solver_config=solver_config or {},
    )
    ctx.transition_to(LoadingState())
    result = ctx.run()

    if verbose:
        _print_result(result)

    return result


def run_batch(
    solver_names: List[str],
    items_path: str,
    budget: float = float("inf"),
    character_level: int = 99,
    time_limit: float = 30.0,
    scoring_config: Optional[ScoringConfig] = None,
    solver_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple solvers on the same problem and return all results.

    Useful for comparing algorithm performance on a single game instance.
    Each solver gets its own ``run_optimization`` call with shared data.

    Args:
        solver_names:   List of algorithm names.
        items_path:     Path to the game data file.
        budget:         Maximum total item cost.
        character_level: Items above this level are excluded.
        time_limit:     Solver wall-clock limit (applied to each solver).
        scoring_config: Shared scoring configuration.
        solver_configs: Per-solver hyperparameter dicts (keyed by name).
        verbose:        If True, print a comparison table at the end.

    Returns:
        Dict mapping ``solver_name → result dict``.
    """
    per_solver_configs = solver_configs or {}
    results: Dict[str, Dict[str, Any]] = {}

    for name in solver_names:
        logger.info("Batch: running solver %s", name)
        result = run_optimization(
            solver_name=name,
            items_path=items_path,
            budget=budget,
            character_level=character_level,
            time_limit=time_limit,
            scoring_config=scoring_config,
            solver_config=per_solver_configs.get(name, {}),
            verbose=False,
        )
        results[name] = result

    if verbose:
        if len(solver_names) > 1:
            _print_comparison(results)
        elif solver_names:
            _print_result(results[solver_names[0]])

    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a single optimization result."""
    build = result.get("build")
    solver = result.get("solver", "?")
    score = result.get("score", 0.0)
    cost = result.get("cost", 0.0)
    budget = result.get("budget", 0.0)
    elapsed = result.get("elapsed", 0.0)
    active_syn: list = result.get("active_synergies", [])

    print(f"\n{'=' * 62}")
    print(f"  RESULT — {solver.upper()} ({elapsed:.2f}s)")
    print(f"{'=' * 62}")
    print(f"\n  Score:  {score:.4f}")
    print(f"  Cost:   {cost:.0f} / {budget:.0f}")
    print(f"  Items:  {result.get('items_equipped', 0)}")

    if build:
        print()
        for slot, item in build.slots.items():
            if item is not None:
                stats_str = ", ".join(f"{k}={v:.0f}" for k, v in item.stats.items())
                print(
                    f"    [{slot.name:>12}] {item.name:<25} "
                    f"({item.rarity.name}, ${item.cost:.0f}) — {stats_str}"
                )
            else:
                print(f"    [{slot.name:>12}] (empty)")

        total: Dict[str, float] = result.get("total_stats", {})
        if total:
            print("\n  Total Stats:")
            for stat, value in sorted(total.items()):
                print(f"    {stat:<22} {value:>8.1f}")

    if active_syn:
        print("\n  Active Synergies:")
        for syn in active_syn:
            print(f"    \u2736 {syn}")

    print(f"\n{'=' * 62}\n")


def _print_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a side-by-side comparison table for a batch run."""
    print(f"\n{'=' * 62}")
    print("  SOLVER COMPARISON")
    print(f"{'=' * 62}")
    print(f"  {'Solver':<15} {'Score':>10} {'Cost':>10} {'Items':>6} {'Time':>8}")
    print(f"  {'-' * 52}")

    for name, res in sorted(results.items(), key=lambda kv: -kv[1].get("score", 0.0)):
        score = res.get("score", 0.0)
        cost = res.get("cost", 0.0)
        items_eq = res.get("items_equipped", 0)
        elapsed = res.get("elapsed", 0.0)
        ok = "\u2713" if res.get("success") else "\u2717"
        print(f"  {name:<14} {ok} {score:>10.4f} {cost:>10.0f} {items_eq:>6} {elapsed:>7.2f}s")

    print(f"{'=' * 62}\n")

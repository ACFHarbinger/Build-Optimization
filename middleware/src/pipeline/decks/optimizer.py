"""
Deck Build Optimization — top-level entry point for deckbuilding games
(e.g. Slay the Spire 2), parallel to pipeline.games.optimizer.

Deliberately lean rather than mirroring pipeline.games's full
Context/State/Action state machine: a deckbuilding solve is a single
subset-selection pass (load cards -> DeckProblem -> solver -> Deck), with
none of the multi-stage branching the games pipeline's state machine earns
its keep on. Output persistence is *not* duplicated -- it reuses
pipeline.games.optimizer's `_persist_run`/`_build_to_result_json` directly,
which work unmodified on a Deck because Deck subclasses Build (see
core.deck.Deck's docstring).

Usage::

    from pipeline.decks import run_deck_optimization

    result = run_deck_optimization(
        solver_name="knapsack",
        items_path="src/data/sample/slay_the_spire_2_ironclad.json",
        max_deck_size=18,
        verbose=True,
    )
    print(f"Score: {result['score']:.2f}")
    print(f"Deck size: {result['items_equipped']}/{result['extra']['max_deck_size']}")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.deck_problem import DeckProblem
from core.scoring import ScoringConfig
from core.synergy import SynergyEngine
from data.datasets.games.file_source import FileSource
from data.transforms import deduplicate

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


@_register("greedy")
def _solve_greedy(problem: DeckProblem, config: Dict[str, Any], time_limit: float) -> Tuple[np.ndarray, float]:
    """Deterministic greedy: take cards in descending value order."""
    selected = problem.greedy_solution()
    score = problem.score_fast(selected)
    return selected, score


@_register("knapsack")
def _solve_knapsack(problem: DeckProblem, config: Dict[str, Any], time_limit: float) -> Tuple[np.ndarray, float]:
    """Exact solve via the C++ backend's plain 0-1 knapsack (backend/src/knapsack.cpp):
    weight=1 per card, capacity=max_deck_size -- the deck-building analogue
    of the `bnb` (MCKP) solver registered for equipment builds, using the
    *other* solver B2 exposed. Falls back to greedy if the backend isn't
    built (see core.native_backend.load_backend).

    Note: with every card's weight fixed at 1, 0-1 knapsack reduces to
    "take the K highest-value items", which greedy-by-value already solves
    exactly -- so this and `_solve_greedy` will always agree on this shape.
    It's kept as the deck-building analogue of `bnb` for architectural
    parity with the equipment pipeline, and as the natural extension point
    once per-card weight (e.g. energy cost) becomes a real constraint
    dimension (see moon/ROADMAP.md V9).
    """
    from core.native_backend import load_backend

    try:
        backend = load_backend()
    except ImportError:
        logger.warning("C++ backend not built; falling back to greedy for the 'knapsack' deck solver")
        return _solve_greedy(problem, config, time_limit)

    items = [backend.KnapsackItem(1, problem.item_value(i)) for i in range(problem.num_items)]
    result = backend.solve_knapsack(items, problem.max_deck_size)

    selected = np.zeros(problem.num_items, dtype=bool)
    for idx in result.selected_indices:
        selected[idx] = True

    score = problem.score_fast(selected)
    return selected, score


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_deck_optimization(
    solver_name: str,
    items_path: str,
    max_deck_size: int,
    budget: float = float("inf"),
    time_limit: float = 30.0,
    scoring_config: Optional[ScoringConfig] = None,
    solver_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    experiment_name: str = "slay-the-spire-2",
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Run one deck-building optimization pass and return the best deck found.

    Args:
        solver_name:    Algorithm to use: ``"greedy"`` or ``"knapsack"``.
        items_path:     Path to the card data file (JSON or CSV).
        max_deck_size:  Maximum number of cards the deck may contain.
        budget:         Maximum total card cost (gold), if constrained; defaults
                         to unbounded since deck *size* is the report's own
                         primary strategic lever, not gold efficiency.
        time_limit:     Solver wall-clock limit in seconds (unused by the
                         current exact/greedy solvers; kept for interface
                         parity with `pipeline.games.run_optimization`).
        scoring_config: Stat weights and bonuses; defaults to ScoringConfig().
        solver_config:  Algorithm hyperparameters (currently unused).
        verbose:        If True, print a formatted result summary.
        experiment_name: Tracking experiment name / `outputs/` subdirectory.
        persist:        If True (default), write a result JSON to `outputs/`
            and log the run to the tracking database, identically to
            `pipeline.games.run_optimization` (reuses its `_persist_run`).

    Returns:
        Result dict with keys: ``success``, ``solver``, ``score``, ``cost``,
        ``budget``, ``items_equipped``, ``elapsed``, ``active_synergies``,
        ``build`` (a core.deck.Deck), ``extra``, ``slots`` (always ``{}`` --
        decks have no slot mapping), ``total_stats``.
    """
    logger.info("Loading card data from %s", items_path)
    source = FileSource(items_path=items_path)
    raw_items = source.fetch_items()
    raw_synergies = source.fetch_synergies()
    items = deduplicate(raw_items)
    logger.info("Loaded %d cards (from %d raw), %d synergy rules", len(items), len(raw_items), len(raw_synergies))

    synergy_engine = SynergyEngine(rules=raw_synergies)
    problem = DeckProblem.from_items(
        items=items,
        max_deck_size=max_deck_size,
        budget=budget,
        scoring_config=scoring_config or ScoringConfig(),
        synergy_engine=synergy_engine,
    )
    logger.info("DeckProblem created: %d cards, target deck size %d", problem.num_items, problem.max_deck_size)

    solver_fn = _SOLVERS.get(solver_name)
    if solver_fn is None:
        raise ValueError(f"Unknown deck solver: {solver_name!r}. Available: {list(_SOLVERS)}")

    start = time.time()
    selected, fast_score = solver_fn(problem, solver_config or {}, time_limit)
    elapsed = time.time() - start
    logger.info(
        "Solver %s finished in %.2fs — score=%.4f, deck size=%d/%d",
        solver_name,
        elapsed,
        fast_score,
        int(selected.sum()),
        max_deck_size,
    )

    deck = problem.to_deck(selected)
    full_score = problem.score_full(selected)
    active_synergies = synergy_engine.active_synergies(deck)

    result: Dict[str, Any] = {
        "success": True,
        "solver": solver_name,
        "score": full_score,
        "cost": deck.total_cost,
        "budget": budget,
        "items_equipped": len(deck),
        "elapsed": elapsed,
        "active_synergies": active_synergies,
        "build": deck,
        "extra": {"fast_score": fast_score, "max_deck_size": max_deck_size},
        "slots": {},
        "total_stats": deck.total_stats,
    }

    if verbose:
        _print_result(result)

    if persist:
        # Reused unmodified: works on Deck because Deck subclasses Build.
        from pipeline.games.optimizer import _persist_run

        _persist_run(
            result,
            experiment_name=experiment_name,
            solver_name=solver_name,
            items_path=items_path,
            budget=budget,
            time_limit=time_limit,
            solver_config=solver_config or {},
            extra_params={"max_deck_size": max_deck_size},
        )

    return result


def _print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a single deck-optimization result."""
    deck = result.get("build")
    solver = result.get("solver", "?")
    score = result.get("score", 0.0)
    elapsed = result.get("elapsed", 0.0)
    max_deck_size = result.get("extra", {}).get("max_deck_size", "?")
    active_syn: list = result.get("active_synergies", [])

    print(f"\n{'=' * 62}")
    print(f"  DECK RESULT — {solver.upper()} ({elapsed:.2f}s)")
    print(f"{'=' * 62}")
    print(f"\n  Score: {score:.4f}")
    print(f"  Size:  {result.get('items_equipped', 0)} / {max_deck_size}")
    print(f"  Cost:  {result.get('cost', 0):.0f}")

    if deck:
        print("\n  Cards:")
        for item in deck.equipped_items:
            stats_str = ", ".join(f"{k}={v:.0f}" for k, v in item.stats.items())
            print(f"    [{item.slot.name:>6}] {item.name:<25} ({item.rarity.name}, ${item.cost:.0f}) — {stats_str}")

        total: Dict[str, float] = result.get("total_stats", {})
        if total:
            print("\n  Total Stats:")
            for stat, value in sorted(total.items()):
                print(f"    {stat:<22} {value:>8.1f}")

    if active_syn:
        print("\n  Active Synergies:")
        for syn in active_syn:
            print(f"    ✶ {syn}")

    print(f"\n{'=' * 62}\n")

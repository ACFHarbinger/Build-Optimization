"""
Pipeline: Deck Build Optimization (deckbuilding games, e.g. Slay the Spire 2).

Parallel to pipeline.games but for variable-size *subset* selection
(core.deck_problem.DeckProblem) rather than one-item-per-slot assignment
(core.problem.BuildProblem) -- see optimizer.py's module docstring and
docs/deck-problem-mapping.md for the full domain mapping.

Usage::

    from pipeline.decks import run_deck_optimization

    result = run_deck_optimization(
        solver_name="knapsack",
        items_path="src/data/sample/slay_the_spire_2_ironclad.json",
        max_deck_size=18,
        verbose=True,
    )
"""

from .optimizer import run_deck_optimization

__all__ = [
    "run_deck_optimization",
]

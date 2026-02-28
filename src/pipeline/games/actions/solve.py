"""
SolveAction: runs the optimization algorithm and writes the best build.

Used by SolvingState (and usable standalone) to execute whichever solver
is named in the context, then store the resulting Build and score.

Supported solvers (dispatched from ``context["solver_name"]``):
    ``"greedy"`` – Deterministic one-pass greedy.
    ``"sa"``     – Simulated Annealing.
    ``"ga"``     – Genetic Algorithm.
"""

import time
from typing import Any, Dict

from .base import GameAction


class SolveAction(GameAction):
    """
    Runs the optimization algorithm named in ``context["solver_name"]``.

    Expected context keys (read):
        ``problem``          – BuildProblem instance.
        ``solver_name``      – Algorithm identifier.
        ``solver_config``    – Hyperparameter dict (optional).
        ``time_limit``       – Wall-clock limit in seconds (optional, default 30).
        ``character_level``  – Used when converting solution to Build.

    Context keys written:
        ``best_build``   – core.build.Build object.
        ``best_score``   – Full score (with synergy bonuses).
        ``extra``        – Dict with ``fast_score`` and ``slot_to_item``.
        ``elapsed``      – Wall-clock seconds taken.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        from pipeline.games.states.solving import _SOLVERS

        problem = context["problem"]
        solver_name: str = context.get("solver_name", "greedy").lower()
        solver_config: Dict[str, Any] = context.get("solver_config", {})
        time_limit: float = float(context.get("time_limit", 30.0))
        character_level: int = int(context.get("character_level", 99))

        solver_fn = _SOLVERS.get(solver_name)
        if solver_fn is None:
            raise ValueError(f"Unknown build solver: {solver_name!r}. Available: {list(_SOLVERS)}")

        start = time.time()
        slot_to_item, fast_score = solver_fn(problem, solver_config, time_limit)
        elapsed = time.time() - start

        context["best_build"] = problem.to_build(slot_to_item, character_level)
        context["best_score"] = problem.score_full(slot_to_item, character_level)
        context["extra"] = {
            "fast_score": fast_score,
            "slot_to_item": slot_to_item.tolist(),
        }
        context["elapsed"] = elapsed

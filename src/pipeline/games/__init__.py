"""
Pipeline: Game Build Optimization.

Provides a clean, state-machine-based pipeline for running optimization
algorithms (Greedy, SA, GA, …) on videogame build problems (RPG, FPS,
MOBA, etc.).

The pipeline mirrors the simulation pipeline in structure:

    Simulation pipeline:          Games pipeline:
    ─────────────────────         ───────────────────────
    SimulationContext         →   GameOptimizationContext
    SimState (ABC)            →   GameState (ABC)
    InitializingState         →   LoadingState
    RunningState              →   SolvingState
    FinishingState            →   ReportingState
    SimulationAction (ABC)    →   GameAction (ABC)
    FillAction, …             →   LoadItemsAction, SolveAction, ReportAction
    single_simulation()       →   run_optimization()
    sequential_simulations()  →   run_batch()

Usage::

    from pipeline.games import run_optimization, run_batch

    result = run_optimization(
        solver_name="sa",
        items_path="src/data/sample/rpg.json",
        budget=1500.0,
        verbose=True,
    )

    results = run_batch(
        solver_names=["greedy", "sa", "ga"],
        items_path="src/data/sample/rpg.json",
        budget=2000.0,
        verbose=True,
    )
"""

from .context import GameOptimizationContext
from .optimizer import run_batch, run_optimization
from .problem import BuildProblem

__all__ = [
    "BuildProblem",
    "GameOptimizationContext",
    "run_optimization",
    "run_batch",
]

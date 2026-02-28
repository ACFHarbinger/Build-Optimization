"""
GameOptimizationContext: central state for a build optimization run.

Analogous to SimulationContext in the simulation pipeline, this object
carries all configuration, data, and results through the state machine
(LoadingState → SolvingState → ReportingState).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.item import Item
from core.problem import BuildProblem
from core.scoring import ScoringConfig
from core.synergy import SynergyEngine

if TYPE_CHECKING:
    from core.build import Build

    from .states.base import GameState


class GameOptimizationContext:
    """
    Context object for the game build optimization state machine.

    Stores configuration and runtime state.  All states access data via
    this object and write their outputs back to it.

    Attributes:
        solver_name:     Algorithm to run (e.g. ``"sa"``, ``"ga"``, ``"greedy"``).
        items_path:      Path to the game data file (JSON or CSV).
        budget:          Maximum total item cost.
        character_level: Items above this level are excluded.
        time_limit:      Solver wall-clock limit in seconds.
        scoring_config:  Stat weights and bonuses.
        solver_config:   Algorithm hyperparameters (flat dict).
        synergy_engine:  Populated during LoadingState.
        items:           Loaded, filtered items (populated during LoadingState).
        problem:         BuildProblem instance (populated during LoadingState).
        best_build:      Best Build found (populated during SolvingState).
        best_score:      Score of the best build (populated during SolvingState).
        extra:           Extra diagnostics from the solver.
        elapsed:         Wall-clock seconds taken by the solver.
        result:          Final result dict (set by ReportingState).
    """

    def __init__(
        self,
        solver_name: str = "greedy",
        items_path: str = "",
        budget: float = float("inf"),
        character_level: int = 99,
        time_limit: float = 30.0,
        scoring_config: Optional[ScoringConfig] = None,
        solver_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Configuration
        self.solver_name = solver_name
        self.items_path = items_path
        self.budget = float(budget)
        self.character_level = int(character_level)
        self.time_limit = float(time_limit)
        self.scoring_config: ScoringConfig = scoring_config or ScoringConfig()
        self.solver_config: Dict[str, Any] = solver_config or {}

        # Domain objects populated by LoadingState
        self.synergy_engine: Optional[SynergyEngine] = None
        self.items: List[Item] = []
        self.problem: Optional[BuildProblem] = None

        # Results populated by SolvingState
        self.best_build: Optional["Build"] = None
        self.best_score: float = 0.0
        self.extra: Dict[str, Any] = {}
        self.elapsed: float = 0.0

        # State machine
        self.current_state: Optional["GameState"] = None
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition_to(self, state: Optional["GameState"]) -> None:
        """Move to the next state."""
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self

    def run(self) -> Dict[str, Any]:
        """Execute the state machine until completion."""
        while self.current_state is not None:
            self.current_state.handle(self)
        return self.result or {}

"""
Build-Optimization: Hydra entry point.

Usage:
    python main.py policy=policy_sa game=rpg
    python main.py policy=policy_ga game=darktide optimization.budget=30000
    python main.py policy=policy_sa game=moba optimization.character_level=30
    python main.py game=darktide output.compare_baselines=true
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# middleware/src holds `core`, `pipeline`, etc. as top-level importable
# packages (not a `middleware.src.*` namespace) — add it to sys.path so this
# entry point works regardless of the invoking working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "middleware" / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from core.scoring import ScoringConfig
from pipeline.decks import run_deck_optimization
from pipeline.games import run_batch, run_optimization

logger = logging.getLogger(__name__)

# Solvers natively supported by the (equipment) games pipeline
_PIPELINE_SOLVERS = {"greedy", "sa", "ga", "bnb"}

# Deck-building games (game.problem_type: "deck") use a separate solver
# namespace/pipeline (pipeline.decks) -- see core.deck_problem.DeckProblem's
# module docstring for why equipment's Multiple-Choice Knapsack model
# doesn't apply to variable-size deck subset selection.
_DECK_POLICY_TO_SOLVER: Dict[str, str] = {
    "deck_knapsack": "knapsack",
    "deck_greedy": "greedy",
}

# Best-effort mapping from policy names to pipeline solver names
_SOLVER_ALIAS: Dict[str, str] = {
    "alns": "sa",
    "gls": "sa",
    "ils": "sa",
    "lahc": "sa",
    "oba": "sa",
    "rts": "sa",
    "rrt": "sa",
    "vns": "sa",
    "sans": "sa",
    "sisr": "sa",
    "hgs": "ga",
    "hgs_alns": "sa",
    "gphh": "ga",
    "neural": "ga",
    "abc": "sa",
    "ahvpl": "sa",
    "bcp": "greedy",
    "fa": "sa",
    "hs": "sa",
    "hvpl": "sa",
    "ks_aco": "sa",
    "hh_aco": "sa",
    "lca": "sa",
    "lkh": "greedy",
    "psoma": "ga",
    "qde": "ga",
    "sca": "sa",
    "slc": "sa",
    "vrpp": "greedy",
    "hmm_gd": "sa",
}


def _build_scoring_config(cfg: DictConfig) -> ScoringConfig:
    """Build ScoringConfig from Hydra game config."""
    scoring = cfg.game.get("scoring", {})
    return ScoringConfig(
        stat_weights=dict(scoring.get("stat_weights", {})),
        synergy_multiplier=scoring.get("synergy_multiplier", 1.5),
        cost_penalty=scoring.get("cost_penalty", 0.0),
        slot_bonus=scoring.get("slot_bonus", 5.0),
        rarity_bonus=scoring.get("rarity_bonus", 2.0),
        diminishing_returns=scoring.get("diminishing_returns", True),
        diminishing_threshold=scoring.get("diminishing_threshold", 200.0),
    )


def _resolve_solver(policy_key: str) -> str:
    """Map a policy config key to a supported pipeline solver name."""
    if policy_key in _PIPELINE_SOLVERS:
        return policy_key
    alias = _SOLVER_ALIAS.get(policy_key)
    if alias:
        return alias
    logger.warning("Unknown policy '%s'; defaulting pipeline solver to 'sa'", policy_key)
    return "sa"


def _flatten_solver_params(raw: Any) -> Dict[str, Any]:
    """Convert policy-yaml params to flat kwargs for the pipeline solver.

    Policy yamls use the format::

        sa:
          custom:
            - initial_temp: 200.0
            - alpha: 0.998

    This flattens the ``custom`` list into ``{"initial_temp": 200.0, ...}``,
    dropping VRP-specific keys (``engine``, ``must_go``, ``post_processing``)
    that the build-optimization pipeline does not use.
    """
    _DROP_KEYS = {"engine", "must_go", "post_processing"}

    if isinstance(raw, list):
        result: Dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict):
                result.update(item)
        return {k: v for k, v in result.items() if k not in _DROP_KEYS}

    if isinstance(raw, dict):
        inner = raw.get("custom", raw)
        if isinstance(inner, list):
            return _flatten_solver_params(inner)
        return {k: v for k, v in inner.items() if k not in _DROP_KEYS}

    return {}


def _run_deck_game(cfg: DictConfig, policy_cfg: DictConfig, scoring_config: ScoringConfig, items_path: str) -> None:
    """Dispatch for deckbuilding games (game.problem_type: "deck"), e.g.
    Slay the Spire 2 -- see core.deck_problem.DeckProblem and
    pipeline.decks.run_deck_optimization."""
    policy_key = list(policy_cfg.keys())[0] if policy_cfg else "deck_knapsack"
    solver_name = _DECK_POLICY_TO_SOLVER.get(policy_key, "knapsack")
    if policy_key not in _DECK_POLICY_TO_SOLVER:
        logger.warning("Unknown deck policy '%s'; defaulting to 'knapsack'", policy_key)

    max_deck_size = OmegaConf.select(cfg, "optimization.max_deck_size", default=18)
    budget = OmegaConf.select(cfg, "optimization.budget", default=float("inf"))
    time_limit = OmegaConf.select(cfg, "optimization.time_limit", default=30.0)
    experiment_name = cfg.game.get("name", "deck-optimization")

    print("=" * 62)
    print("  BUILD OPTIMIZATION — Deckbuilding")
    print("=" * 62)
    print(f"\n  Game:      {cfg.game.get('name', 'Unknown')}")
    print(f"  Solver:    {policy_key} → {solver_name}")
    print(f"  Deck size: {max_deck_size}")
    print(f"  Budget:    {budget:,.0f}" if budget != float("inf") else "  Budget:    inf")
    print(f"  Time:      {time_limit}s")
    print()

    result = run_deck_optimization(
        solver_name=solver_name,
        items_path=items_path,
        max_deck_size=max_deck_size,
        budget=budget,
        time_limit=time_limit,
        scoring_config=scoring_config,
        verbose=True,
        experiment_name=experiment_name,
    )
    if not result.get("success"):
        print("  WARNING: No feasible deck found within size/budget constraints.")
        sys.exit(1)


@hydra.main(config_path="middleware/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for build optimization."""
    scoring_config = _build_scoring_config(cfg)
    items_path = cfg.game.data.items_path
    policy_cfg = cfg.get("policy", {})

    if cfg.game.get("problem_type", "build") == "deck":
        _run_deck_game(cfg, policy_cfg, scoring_config, items_path)
        return

    # Extract solver identity from the loaded policy config
    policy_key = list(policy_cfg.keys())[0] if policy_cfg else "sa"
    solver_name = _resolve_solver(policy_key)

    solver_label = (
        f"{policy_key} → {solver_name}" if solver_name != policy_key else solver_name
    )

    print("=" * 62)
    print("  BUILD OPTIMIZATION — Videogame Build Optimizer")
    print("=" * 62)
    print(f"\n  Game:   {cfg.game.get('name', 'Unknown')}")
    print(f"  Solver: {solver_label}")
    print(f"  Budget: {cfg.optimization.budget:,.0f}")
    print(f"  Level:  {cfg.optimization.character_level}")
    print(f"  Time:   {cfg.optimization.time_limit}s")
    print()

    raw_params = OmegaConf.to_container(policy_cfg.get(policy_key, {}), resolve=True)
    solver_kwargs = _flatten_solver_params(raw_params)

    compare = cfg.output.get("compare_baselines", False)
    experiment_name = cfg.game.get("name", "build-optimization")

    if compare:
        print("  Comparing solvers: greedy / sa / ga")
        run_batch(
            solver_names=["greedy", "sa", "ga"],
            items_path=items_path,
            budget=cfg.optimization.budget,
            character_level=cfg.optimization.character_level,
            time_limit=cfg.optimization.time_limit,
            scoring_config=scoring_config,
            solver_configs={solver_name: solver_kwargs},
            verbose=True,
            experiment_name=experiment_name,
        )
    else:
        result = run_optimization(
            solver_name=solver_name,
            items_path=items_path,
            budget=cfg.optimization.budget,
            character_level=cfg.optimization.character_level,
            time_limit=cfg.optimization.time_limit,
            scoring_config=scoring_config,
            solver_config=solver_kwargs,
            verbose=True,
            experiment_name=experiment_name,
        )
        if not result.get("success"):
            print("  WARNING: No feasible build found within budget constraints.")
            sys.exit(1)


if __name__ == "__main__":
    main()

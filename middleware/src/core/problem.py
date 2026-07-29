"""
BuildProblem: Problem definition for game build optimization.

Bridges the data layer (items, synergies, scoring config) and the
optimization algorithms (SA, GA, greedy, etc.).  Items are stored as
parallel numpy arrays for efficient vectorised solver operations, while
the original Item objects are retained for full synergy-aware scoring.

Solution representation used by solvers:
    ``slot_to_item`` — a 1-D int array of length ``num_slots`` where
    ``slot_to_item[s]`` is the global item index assigned to slot ``s``,
    or ``-1`` for an empty slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from core.item import Item, Slot
from core.scoring import ScoringConfig, score_build
from core.synergy import SynergyEngine

if TYPE_CHECKING:
    from core.build import Build


class BuildProblem:
    """
    A single build-optimisation instance, independent of PyTorch/TorchRL.

    Attributes:
        stat_matrix:    [N, S] normalised item stats (each column in [0, 1]).
        costs:          [N] item acquisition costs.
        slot_ids:       [N] integer slot index for each item.
        rarities:       [N] rarity multiplier per item.
        stat_weights:   [S] per-stat scoring weights from ScoringConfig.
        budget:         Maximum total cost allowed.
        slot_bonus:     Score bonus per filled equipment slot.
        rarity_bonus:   Score bonus per rarity multiplier unit.
        num_slots:      Number of distinct equipment slots.
        stat_names:     Stat names aligned to stat_matrix columns.
        items:          Original Item objects (needed for synergy scoring).
        scoring_config: ScoringConfig stored for reference.
        synergy_engine: Optional SynergyEngine for synergy bonuses.
    """

    def __init__(
        self,
        stat_matrix: np.ndarray,
        costs: np.ndarray,
        slot_ids: np.ndarray,
        rarities: np.ndarray,
        stat_weights: np.ndarray,
        budget: float,
        slot_bonus: float = 5.0,
        rarity_bonus: float = 2.0,
        num_slots: int = 10,
        stat_names: Optional[List[str]] = None,
        items: Optional[List[Item]] = None,
        scoring_config: Optional[ScoringConfig] = None,
        synergy_engine: Optional[SynergyEngine] = None,
    ) -> None:
        assert stat_matrix.ndim == 2, "stat_matrix must be 2-D [N, S]"
        assert costs.shape == (stat_matrix.shape[0],)
        assert slot_ids.shape == (stat_matrix.shape[0],)
        assert rarities.shape == (stat_matrix.shape[0],)
        assert stat_weights.shape == (stat_matrix.shape[1],)

        self.stat_matrix = stat_matrix
        self.costs = costs
        self.slot_ids = slot_ids
        self.rarities = rarities
        self.stat_weights = stat_weights
        self.budget = float(budget)
        self.slot_bonus = float(slot_bonus)
        self.rarity_bonus = float(rarity_bonus)
        self.num_slots = int(num_slots)
        self.stat_names: List[str] = stat_names or []
        self.items: List[Item] = items or []
        self.scoring_config = scoring_config
        self.synergy_engine = synergy_engine
        self._slots: List[Slot] = list(Slot)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_items(self) -> int:
        return int(self.costs.shape[0])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_items(
        cls,
        items: List[Item],
        budget: float,
        scoring_config: Optional[ScoringConfig] = None,
        synergy_engine: Optional[SynergyEngine] = None,
    ) -> "BuildProblem":
        """
        Construct a BuildProblem from core domain objects.

        Derives the stat universe from the ScoringConfig and any extra
        stats found on items, normalises stat values to [0, 1] per column,
        and builds all parallel numpy arrays.

        Args:
            items:          Items to optimise over (post-filter).
            budget:         Maximum allowed total cost.
            scoring_config: Scoring weights and bonuses (defaults to ScoringConfig()).
            synergy_engine: Optional synergy engine for bonus scoring.

        Returns:
            A fully populated BuildProblem ready for solvers.
        """
        cfg = scoring_config or ScoringConfig()
        slots_enum = list(Slot)
        slot_map: Dict[Slot, int] = {slot: i for i, slot in enumerate(slots_enum)}
        num_slots = len(slots_enum)

        # Build stat universe: config-declared stats first, then any extras
        stat_names: List[str] = list(cfg.stat_weights.keys())
        for it in items:
            for s in it.stats:
                if s not in stat_names:
                    stat_names.append(s)

        S = len(stat_names)
        N = len(items)

        # Raw stat matrix [N, S]
        raw = np.zeros((N, S), dtype=np.float32)
        for row, it in enumerate(items):
            for col, sname in enumerate(stat_names):
                raw[row, col] = float(it.stats.get(sname, 0.0))

        # Normalise columns to [0, 1]
        col_max = raw.max(axis=0)
        col_max[col_max == 0.0] = 1.0
        stat_matrix = raw / col_max

        stat_weights = np.array([cfg.stat_weights.get(s, 1.0) for s in stat_names], dtype=np.float32)
        costs = np.array([float(it.cost) for it in items], dtype=np.float32)
        slot_ids = np.array([slot_map[it.slot] for it in items], dtype=np.int64)
        rarities = np.array([it.rarity.multiplier for it in items], dtype=np.float32)

        return cls(
            stat_matrix=stat_matrix,
            costs=costs,
            slot_ids=slot_ids,
            rarities=rarities,
            stat_weights=stat_weights,
            budget=float(budget),
            slot_bonus=float(cfg.slot_bonus),
            rarity_bonus=float(cfg.rarity_bonus),
            num_slots=num_slots,
            stat_names=stat_names,
            items=items,
            scoring_config=cfg,
            synergy_engine=synergy_engine,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def evaluate(self, slot_to_item: np.ndarray) -> float:
        """
        Alias for score_fast, compatible with ALNS/ACO solvers.
        """
        return self.score_fast(slot_to_item)

    def score_fast(self, slot_to_item: np.ndarray) -> float:
        """
        Fast approximate score for a slot-to-item assignment.

        Skips synergy bonuses (which require a Build object) but is
        suitable for use inside tight optimiser loops.

        Args:
            slot_to_item: [num_slots] int array; -1 means empty slot.

        Returns:
            Scalar score (float).
        """
        score = 0.0
        for item_idx in slot_to_item:
            if item_idx < 0:
                continue
            score += float(np.dot(self.stat_matrix[item_idx], self.stat_weights))
            score += self.slot_bonus
            score += float(self.rarities[item_idx]) * self.rarity_bonus
        return score

    def score_full(self, slot_to_item: np.ndarray, character_level: int = 99) -> float:
        """
        Full score via core.scoring (includes synergy bonuses).

        Converts the solver array to a Build object and calls score_build().

        Args:
            slot_to_item:    [num_slots] int array; -1 means empty slot.
            character_level: Character level for the Build.

        Returns:
            Scalar score (float).
        """
        build = self.to_build(slot_to_item, character_level)
        cfg = self.scoring_config or ScoringConfig()
        return score_build(build, self.synergy_engine, cfg)

    # ------------------------------------------------------------------
    # Solution helpers
    # ------------------------------------------------------------------

    def is_feasible(self, slot_to_item: np.ndarray) -> bool:
        """True if total cost of assigned items ≤ budget."""
        return self.budget_used(slot_to_item) <= self.budget

    def budget_used(self, slot_to_item: np.ndarray) -> float:
        """Total cost of all assigned items."""
        mask = slot_to_item >= 0
        if not mask.any():
            return 0.0
        return float(self.costs[slot_to_item[mask]].sum())

    def to_build(self, slot_to_item: np.ndarray, character_level: int = 99) -> "Build":
        """
        Convert a solver's slot-to-item array to a core.build.Build.

        Args:
            slot_to_item:    [num_slots] int array; -1 means empty slot.
            character_level: Character level constraint for the Build.

        Returns:
            A Build with all valid items equipped.
        """
        from core.build import Build

        build = Build(budget=self.budget, character_level=character_level)
        for item_idx in slot_to_item:
            if item_idx >= 0 and item_idx < len(self.items):
                build.equip(self.items[item_idx])
        return build

    # ------------------------------------------------------------------
    # Initial solutions
    # ------------------------------------------------------------------

    def greedy_solution(self) -> np.ndarray:
        """
        Deterministic greedy initial solution.

        For each slot (in order), selects the affordable item with the
        highest weighted-stat + rarity score.

        Returns:
            [num_slots] int array (item index per slot, or -1).
        """
        solution = np.full(self.num_slots, -1, dtype=np.int64)
        remaining = self.budget
        score_vec = (self.stat_matrix @ self.stat_weights) + self.rarities * self.rarity_bonus

        for slot_idx in range(self.num_slots):
            candidates = np.where(self.slot_ids == slot_idx)[0]
            if len(candidates) == 0:
                continue
            affordable = candidates[self.costs[candidates] <= remaining]
            if len(affordable) == 0:
                continue
            best = affordable[int(np.argmax(score_vec[affordable]))]
            solution[slot_idx] = int(best)
            remaining -= float(self.costs[best])

        return solution

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Random feasible initial solution.

        Randomly assigns one affordable item per slot in a randomised
        slot order.

        Args:
            rng: NumPy random generator (uses default_rng() if None).

        Returns:
            [num_slots] int array (item index per slot, or -1).
        """
        if rng is None:
            rng = np.random.default_rng()
        solution = np.full(self.num_slots, -1, dtype=np.int64)
        remaining = self.budget

        for slot_idx in rng.permutation(self.num_slots):
            candidates = np.where(self.slot_ids == slot_idx)[0]
            if len(candidates) == 0:
                continue
            affordable = candidates[self.costs[candidates] <= remaining]
            if len(affordable) == 0:
                continue
            chosen = int(rng.choice(affordable))
            solution[int(slot_idx)] = chosen
            remaining -= float(self.costs[chosen])

        return solution

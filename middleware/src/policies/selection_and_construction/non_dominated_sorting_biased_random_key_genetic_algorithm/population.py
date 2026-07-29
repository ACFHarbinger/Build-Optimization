"""
NDS-BRKGA Population Management Module.

Manages the full population lifecycle for the Non-Dominated Sorting Biased
Random-Key Genetic Algorithm: initialisation, seeding, and generational
breeding.

Domain-Knowledge Seeding
------------------------
The BRKGA's "biased" claim rests in part on seeding the initial population
with near-optimal solutions derived from individual sub-problem solvers:

* **Selection seeder** (default: ``FractionalKnapsackSelection``):
  Solves the bin-selection sub-problem in isolation.  The resulting
  selected/unselected partition is encoded into selection keys.

* **Routing seeder** (default: greedy distance-nearest insertion):
  Given a fixed set of selected bins, builds a greedy tour and encodes
  the visit order into routing keys.

Cross-seeded chromosomes (selection from solver A, routing independently
optimised) are also injected to capture complementary structure.

The remaining population slots are filled with uniformly random chromosomes
to maintain genetic diversity.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    Chromosome,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.crossover import (
    biased_crossover,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2 import (
    select_elite_nsga2,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives import (
    evaluate_population,
)
from logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params import (
    NDSBRKGAParams,
)

# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _seed_selection(
    n_bins: int,
    current_fill: np.ndarray,
    distance_matrix: np.ndarray,
    capacity: float,
    revenue_kg: float,
    bin_density: float,
    bin_volume: float,
    max_fill: float,
    overflow_penalty_frac: float,
    scenario_tree: Optional[object],
    strategy_name: str,
    rng: np.random.Generator,
) -> List[int]:
    """
    Run a selection sub-problem solver and return selected 1-based bin IDs.

    Falls back to a fill-threshold heuristic (bins >= 80 %) if the
    requested strategy is unavailable or raises an exception.

    Args:
        n_bins: Total number of candidate bins.
        current_fill: Current fill percentages. Shape ``(n_bins,)``.
        distance_matrix: Full distance matrix.
        capacity: Vehicle capacity.
        revenue_kg: Revenue per kg.
        bin_density: Waste density kg/L.
        bin_volume: Bin volume in litres.
        max_fill: Maximum fill level (normally 100.0).
        overflow_penalty_frac: Overflow penalty fraction.
        scenario_tree: Optional ScenarioTree for stochastic risk.
        strategy_name: Name registered in MandatorySelectionFactory.
        rng: NumPy Generator (unused here but passed for API consistency).

    Returns:
        Sorted list of 1-based bin IDs selected by the strategy.
    """
    try:
        from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext
        from logic.src.policies.mandatory_selection.base.selection_factory import MandatorySelectionFactory

        ctx = SelectionContext(
            bin_ids=np.arange(1, n_bins + 1, dtype=np.int32),
            current_fill=current_fill.copy(),
            distance_matrix=distance_matrix,
            vehicle_capacity=capacity,
            revenue_kg=revenue_kg,
            bin_density=bin_density,
            bin_volume=bin_volume,
            max_fill=max_fill,
            overflow_penalty_frac=overflow_penalty_frac,
            scenario_tree=scenario_tree,
            n_vehicles=1,
        )
        strategy = MandatorySelectionFactory.create_strategy(strategy_name)
        selected, _ = strategy.select_bins(ctx)
        return sorted(selected)
    except Exception:
        # Fallback: threshold heuristic
        return sorted((np.nonzero(current_fill >= 80.0)[0] + 1).tolist())


def _seed_routing_order(
    selected_1based: List[int],
    dist_matrix: np.ndarray,
) -> List[int]:
    """
    Build a nearest-neighbour greedy tour order for the selected bins.

    Args:
        selected_1based: 1-based IDs of selected bins.
        dist_matrix: Full distance matrix (row/col 0 = depot).

    Returns:
        The selected bin IDs reordered by nearest-neighbour insertion
        starting from the depot.
    """
    if not selected_1based:
        return []
    remaining = list(selected_1based)
    order: List[int] = []
    current = 0  # depot index
    while remaining:
        nearest = min(remaining, key=lambda b: dist_matrix[current, b])
        order.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return order


def _build_seed_chromosomes(
    n_bins: int,
    current_fill: np.ndarray,
    distance_matrix: np.ndarray,
    capacity: float,
    revenue_kg: float,
    bin_density: float,
    bin_volume: float,
    max_fill: float,
    overflow_penalty_frac: float,
    scenario_tree: Optional[object],
    params: NDSBRKGAParams,
    rng: np.random.Generator,
) -> List[Chromosome]:
    """
    Generate a batch of seeded chromosomes using domain-specific solvers.

    Produces up to ``2 * params.n_seed_solutions`` chromosomes:
    - ``n_seed_solutions`` with the selection from the selection-seeder
      and routing from the greedy nearest-neighbour heuristic.
    - ``n_seed_solutions`` with the selection from a fill-threshold
      heuristic and routing from the greedy nearest-neighbour heuristic.

    Args:
        n_bins: Number of candidate bins.
        current_fill: Current bin fill levels as percentages.
        distance_matrix: Full distance matrix.
        capacity: Vehicle capacity.
        revenue_kg: Revenue per kg.
        bin_density: Waste density kg/L.
        bin_volume: Bin volume in litres.
        max_fill: Maximum fill level.
        overflow_penalty_frac: Overflow penalty fraction.
        scenario_tree: Optional stochastic scenario tree.
        params: NDS-BRKGA hyperparameter configuration.
        rng: NumPy Random Generator.

    Returns:
        List of seeded :class:`~.chromosome.Chromosome` objects.
    """
    seeds: List[Chromosome] = []

    # --- Strategy A: primary selection seeder ---
    sel_a = _seed_selection(
        n_bins,
        current_fill,
        distance_matrix,
        capacity,
        revenue_kg,
        bin_density,
        bin_volume,
        max_fill,
        overflow_penalty_frac,
        scenario_tree,
        params.seed_selection_strategy,
        rng,
    )
    for _ in range(params.n_seed_solutions):
        # Add jitter by randomly adding/dropping one optional bin
        sel_variant = list(sel_a)
        if rng.random() < 0.3 and len(sel_variant) > 1:
            sel_variant.pop(rng.integers(0, len(sel_variant)))
        elif rng.random() < 0.3:
            extra = rng.integers(1, n_bins + 1)
            if extra not in sel_variant:
                sel_variant.append(int(extra))

        routing = _seed_routing_order(sel_variant, distance_matrix)
        chrom = Chromosome.from_selection_and_order(
            n_bins=n_bins,
            selected_bins_0idx=[b - 1 for b in sel_variant],
            routing_order_0idx=[b - 1 for b in routing],
            rng=rng,
        )
        seeds.append(chrom)

    # --- Strategy B: high-fill threshold (backup diversity) ---
    thresholds_fill = [75.0, 85.0, 90.0, 95.0, 99.0]
    for thr in thresholds_fill[: params.n_seed_solutions]:
        sel_b = sorted((np.nonzero(current_fill >= thr)[0] + 1).tolist())
        if not sel_b:
            continue
        routing_b = _seed_routing_order(sel_b, distance_matrix)
        chrom_b = Chromosome.from_selection_and_order(
            n_bins=n_bins,
            selected_bins_0idx=[b - 1 for b in sel_b],
            routing_order_0idx=[b - 1 for b in routing_b],
            rng=rng,
        )
        seeds.append(chrom_b)

    return seeds


# ---------------------------------------------------------------------------
# Population class
# ---------------------------------------------------------------------------


class Population:
    """
    Manages the NDS-BRKGA chromosome pool for a single run.

    Attributes:
        chromosomes: Current list of :class:`~.chromosome.Chromosome` objects.
        objectives: Objective matrix, shape ``(P, 3)``.  Updated after each
            evaluation step.  Columns: ``[neg_profit, overflow_cost, distance]``.
        front_ranks: 1-based Pareto ranks. Shape ``(P,)``.
    """

    def __init__(
        self,
        chromosomes: List[Chromosome],
        objectives: np.ndarray,
        front_ranks: np.ndarray,
    ) -> None:
        self.chromosomes = chromosomes
        self.objectives = objectives
        self.front_ranks = front_ranks

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def initialise(
        cls,
        n_bins: int,
        thresholds: np.ndarray,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        overflow_risk: np.ndarray,
        current_fill: np.ndarray,
        bin_density: float,
        bin_volume: float,
        max_fill: float,
        overflow_penalty_frac: float,
        scenario_tree: Optional[object],
        params: NDSBRKGAParams,
        mandatory_override: Optional[List[int]] = None,
    ) -> "Population":
        """
        Build the initial population: seeded near-optimal + random fill.

        Args:
            n_bins: Number of candidate bins.
            thresholds: Per-bin adaptive selection thresholds. Shape ``(N,)``.
            dist_matrix: Full distance matrix.
            wastes: Fill-percentage mapping for evaluation.
            capacity: Vehicle capacity.
            R: Revenue per fill unit.
            C: Cost per distance unit.
            overflow_risk: Per-bin overflow risk scores.
            current_fill: Current fill levels as percentages.
            bin_density: Waste density kg/L.
            bin_volume: Bin volume in litres.
            max_fill: Maximum fill percentage.
            overflow_penalty_frac: Overflow penalty fraction.
            scenario_tree: Optional stochastic scenario tree.
            params: NDS-BRKGA parameters.
            mandatory_override: 1-based bin IDs that must be visited.

        Returns:
            An initialised and evaluated :class:`Population`.
        """
        rng = np.random.default_rng(params.seed)

        # 1. Generate seeded chromosomes
        seeds = _build_seed_chromosomes(
            n_bins=n_bins,
            current_fill=current_fill,
            distance_matrix=dist_matrix,
            capacity=capacity,
            revenue_kg=R,
            bin_density=bin_density,
            bin_volume=bin_volume,
            max_fill=max_fill,
            overflow_penalty_frac=overflow_penalty_frac,
            scenario_tree=scenario_tree,
            params=params,
            rng=rng,
        )

        # 2. Fill remaining slots with random chromosomes
        n_random = max(0, params.pop_size - len(seeds))
        randoms = [Chromosome.random(n_bins, rng) for _ in range(n_random)]
        chromosomes = seeds[: params.pop_size] + randoms
        chromosomes = chromosomes[: params.pop_size]  # enforce hard cap

        # 3. Evaluate
        objectives = evaluate_population(
            chromosomes,
            thresholds,
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            overflow_risk,
            params.overflow_penalty,
            mandatory_override,
        )

        # 4. NSGA-II initial sort
        _, front_ranks = select_elite_nsga2(objectives, params.n_elite)

        return cls(chromosomes, objectives, front_ranks)

    # ------------------------------------------------------------------
    # Elite extraction
    # ------------------------------------------------------------------

    def get_elite_indices(self, n_elite: int) -> List[int]:
        """
        Return the indices of the top *n_elite* chromosomes.

        Uses NSGA-II rank then crowding-distance tiebreak.

        Args:
            n_elite: Number of elite individuals to return.

        Returns:
            Sorted list of population indices.
        """
        elite_idx, _ = select_elite_nsga2(self.objectives, n_elite)
        return elite_idx

    # ------------------------------------------------------------------
    # Breeding
    # ------------------------------------------------------------------

    def breed_next_generation(
        self,
        n_elite: int,
        n_mutants: int,
        bias_elite: float,
        rng: np.random.Generator,
    ) -> List[Chromosome]:
        """
        Produce a new generation of chromosomes.

        The new generation consists of:
        1. **Elite**: The top ``n_elite`` chromosomes copied unchanged.
        2. **Offspring**: ``pop_size - n_elite - n_mutants`` children
           produced by biased crossover (elite × random non-elite parent).
        3. **Mutants**: ``n_mutants`` fully random chromosomes.

        Args:
            n_elite: Number of chromosomes to preserve unchanged.
            n_mutants: Number of random mutant chromosomes to inject.
            bias_elite: Elite-parent bias for crossover.
            rng: NumPy Random Generator.

        Returns:
            New population list (not yet evaluated).
        """
        pop_size = len(self.chromosomes)
        n_bins = self.chromosomes[0].n_bins

        elite_idx = self.get_elite_indices(n_elite)
        elite_pool = [self.chromosomes[i] for i in elite_idx]
        non_elite_pool = [self.chromosomes[i] for i in range(pop_size) if i not in set(elite_idx)]

        new_generation: List[Chromosome] = list(elite_pool)

        # Offspring via biased crossover
        n_offspring = pop_size - n_elite - n_mutants
        for _ in range(max(0, n_offspring)):
            e_idx = int(rng.integers(0, len(elite_pool)))
            if non_elite_pool:
                ne_idx = int(rng.integers(0, len(non_elite_pool)))
                child = biased_crossover(elite_pool[e_idx], non_elite_pool[ne_idx], bias_elite, rng)
            else:
                # Fallback: two elite parents if no non-elite pool
                e_idx2 = int(rng.integers(0, len(elite_pool)))
                child = biased_crossover(elite_pool[e_idx], elite_pool[e_idx2], bias_elite, rng)
            new_generation.append(child)

        # Mutants
        for _ in range(n_mutants):
            new_generation.append(Chromosome.random(n_bins, rng))

        return new_generation

    # ------------------------------------------------------------------
    # Best solution extraction
    # ------------------------------------------------------------------

    def best_chromosome(self) -> Tuple[Chromosome, np.ndarray]:
        """
        Return the chromosome with the highest net profit (lowest ``neg_profit``).

        Ties are broken by lowest overflow cost, then lowest distance.

        Returns:
            Tuple of ``(chromosome, objective_vector)``.
        """
        # Sort by (neg_profit, overflow_cost, distance) ascending
        priority = np.lexsort(
            (
                self.objectives[:, 2],
                self.objectives[:, 1],
                self.objectives[:, 0],
            )
        )
        best_idx = int(priority[0])
        return self.chromosomes[best_idx], self.objectives[best_idx]

"""
NDS-BRKGA Hyperparameter Configuration.

This module defines :class:`NDSBRKGAParams`, the configuration dataclass
for the Non-Dominated Sorting Biased Random-Key Genetic Algorithm.  All
fields have sensible defaults for the WCVRP/VRPP problem family.

References:
    Gonçalves, J. F., & Resende, M. G. (2011).
        Biased random-key genetic algorithms for combinatorial optimization.
        *Journal of Heuristics*, 17(5), 487–525.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NDSBRKGAParams:
    """
    Configuration for the NDS-BRKGA joint selection-and-construction solver.

    Attributes:
        pop_size: Total population size per generation.
        n_elite: Number of elite chromosomes carried forward via NSGA-II
            non-dominated sorting and crowding-distance tiebreak.
        n_mutants: Number of completely random (mutant) chromosomes injected
            each generation to maintain diversity.
        bias_elite: Probability that a gene in the offspring is inherited
            from the elite parent during biased uniform crossover.  Must be
            in ``(0.5, 1.0)`` to favour the elite parent.
        max_generations: Maximum number of evolutionary generations.  The
            algorithm also respects ``time_limit`` and stops at whichever
            comes first.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.  ``None`` uses the system RNG.
        vrpp: If ``True``, bins beyond the mandatory set are considered as
            optional nodes (VRPP mode).  If ``False``, only selected bins
            are routed (pure selection mode).
        overflow_penalty: Weight applied to the overflow-cost objective
            (unselected bins' expected waste overflow in kg).  Higher values
            push the Pareto front towards safety-first solutions.

        seed_selection_strategy: Name of the
            :class:`~logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy`
            used to seed the initial population with near-optimal selection
            decisions.  Default uses the fractional-knapsack heuristic.
        seed_routing_strategy: Name of the route-construction operator used
            to build the routing component of seed chromosomes.  Can be any
            key registered in
            :class:`~logic.src.policies.route_construction.base.registry.RouteConstructorRegistry`,
            or the special value ``"greedy"`` for the built-in greedy initialiser.
        n_seed_solutions: Number of seed solutions injected per sub-problem
            solver (selection + routing independently, then crossed).
            Total seeded slots = ``min(2 * n_seed_solutions, pop_size // 2)``.

        selection_threshold_min: Lower bound for the per-bin adaptive
            selection threshold.  Bins with maximum overflow risk will use
            this threshold, making them very likely to be selected.
        selection_threshold_max: Upper bound for the per-bin adaptive
            selection threshold.  Bins with zero overflow risk will use
            this threshold, making them very unlikely to be selected.
    """

    pop_size: int = 60
    n_elite: int = 15
    n_mutants: int = 10
    bias_elite: float = 0.70
    max_generations: int = 200
    time_limit: float = 90.0
    seed: Optional[int] = 42
    vrpp: bool = True
    overflow_penalty: float = 10.0

    # Seeding configuration
    seed_selection_strategy: str = "fractional_knapsack"
    seed_routing_strategy: str = "greedy"
    n_seed_solutions: int = 5

    # Adaptive threshold bounds
    selection_threshold_min: float = 0.10
    selection_threshold_max: float = 0.90

    @classmethod
    def from_config(cls, config: object) -> "NDSBRKGAParams":
        """
        Construct from a configuration object (dataclass or dict-like).

        Args:
            config: Any object with attribute or dict-key access.

        Returns:
            NDSBRKGAParams: Populated parameter instance.
        """

        def _get(attr: str, default):  # type: ignore[no-untyped-def]
            if isinstance(config, dict):
                return config.get(attr, default)
            return getattr(config, attr, default)

        return cls(
            pop_size=int(_get("pop_size", 60)),
            n_elite=int(_get("n_elite", 15)),
            n_mutants=int(_get("n_mutants", 10)),
            bias_elite=float(_get("bias_elite", 0.70)),
            max_generations=int(_get("max_generations", 200)),
            time_limit=float(_get("time_limit", 90.0)),
            seed=_get("seed", 42),
            vrpp=bool(_get("vrpp", True)),
            overflow_penalty=float(_get("overflow_penalty", 10.0)),
            seed_selection_strategy=str(_get("seed_selection_strategy", "fractional_knapsack")),
            seed_routing_strategy=str(_get("seed_routing_strategy", "greedy")),
            n_seed_solutions=int(_get("n_seed_solutions", 5)),
            selection_threshold_min=float(_get("selection_threshold_min", 0.10)),
            selection_threshold_max=float(_get("selection_threshold_max", 0.90)),
        )

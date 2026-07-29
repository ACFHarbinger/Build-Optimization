from enum import Enum, auto


class OperatorTag(Enum):
    # Functional Phase
    DESTRUCTIVE = auto()  # Ruin (e.g., Random, Shaw, Worst)
    CONSTRUCTIVE = auto()  # Repair/Recreate (e.g., Greedy, Regret-K)
    IMPROVEMENT = auto()  # Local Search (e.g., 2-Opt, Relocate)
    RECOMBINATION = auto()  # Crossover (e.g., OX, Edge Recombination)

    # Topology Scope
    INTRA_ROUTE = auto()  # Operates within a single route
    INTER_ROUTE = auto()  # Swaps between different routes

    # Computational Complexity
    O_N = auto()  # Linear time
    O_N2 = auto()  # Quadratic (standard for most edge-exchanges)
    O_N3 = auto()  # Cubic (e.g., full 3-Opt)

    # Mathematical Rigor
    HEURISTIC = auto()  # Fast approximation
    EXACT_SUBPROBLEM = auto()  # Solves a sub-graph optimally (DP or Set Partitioning)

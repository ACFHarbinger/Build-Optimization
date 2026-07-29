"""
NSGA-II Non-Dominated Sorting and Crowding Distance Module.

Implements the fast non-dominated sorting algorithm and crowding-distance
assignment from Deb et al. (2002) for extracting an elite subset from the
NDS-BRKGA population.

Algorithm
---------
1.  **Fast Non-Dominated Sort**:  Partition the population into fronts
    ``F₁, F₂, …`` such that every solution in ``Fₖ`` is dominated by at
    least one solution in ``Fₖ₋₁``.  Complexity ``O(M × N²)`` where ``M``
    is the number of objectives and ``N`` is population size.

2.  **Crowding Distance**:  For each solution on a given front, compute the
    average side-length of the cuboid formed by its two neighbours in each
    objective dimension.  Boundary solutions receive distance ``∞``.

3.  **Elite Extraction**:  Fill the elite pool front-by-front.  When the
    next complete front would overflow the budget, rank its members by
    crowding distance (larger = more isolated = more diverse = preferred).

References:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
"""

from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fast non-dominated sort
# ---------------------------------------------------------------------------


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Partition population into Pareto fronts F₁, F₂, … .

    A solution :math:`p` dominates :math:`q` if it is no worse in all
    objectives and strictly better in at least one (all objectives are
    assumed to be minimised).

    Args:
        objectives: Objective matrix of shape ``(P, M)``.  Each row is one
            solution; each column is one objective value to minimise.

    Returns:
        List of fronts.  ``fronts[0]`` is the Pareto-optimal set (rank 1);
        each subsequent element is the next non-dominated layer.  Indices
        refer to rows in *objectives*.
    """
    P = len(objectives)
    domination_count = np.zeros(P, dtype=int)  # how many solutions dominate p
    dominated_by: List[List[int]] = [[] for _ in range(P)]

    for i in range(P):
        for j in range(i + 1, P):
            # Check dominance between i and j
            diff_ij = objectives[i] - objectives[j]  # >0 means i worse than j
            i_dominates_j = bool(np.all(diff_ij <= 0) and np.any(diff_ij < 0))
            j_dominates_i = bool(np.all(diff_ij >= 0) and np.any(diff_ij > 0))

            if i_dominates_j:
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif j_dominates_i:
                dominated_by[j].append(i)
                domination_count[i] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(P) if domination_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for p in current_front:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------


def crowding_distance(front: List[int], objectives: np.ndarray) -> np.ndarray:
    """
    Assign crowding distances to members of a single Pareto front.

    Boundary solutions (best and worst in any objective) receive distance
    ``+∞``.  Interior solutions receive the normalised sum of cuboid
    side-lengths across all objective dimensions.

    Args:
        front: Indices (into *objectives*) of solutions on this front.
        objectives: Full objective matrix, shape ``(P, M)``.

    Returns:
        np.ndarray: Crowding distances for each member of *front*.
            Shape ``(|front|,)``.  Boundary members are ``np.inf``.
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    M = objectives.shape[1]
    distances = np.zeros(n, dtype=float)
    front_obj = objectives[front]  # shape (n, M)

    for m in range(M):
        order = np.argsort(front_obj[:, m])
        f_min = front_obj[order[0], m]
        f_max = front_obj[order[-1], m]
        span = f_max - f_min

        # Boundary members get infinite distance
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if span < 1e-12:
            # All solutions identical in this objective: contribute 0
            continue

        for k in range(1, n - 1):
            distances[order[k]] += (front_obj[order[k + 1], m] - front_obj[order[k - 1], m]) / span

    return distances


# ---------------------------------------------------------------------------
# Elite extraction
# ---------------------------------------------------------------------------


def select_elite_nsga2(
    objectives: np.ndarray,
    n_elite: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Select *n_elite* solutions using NSGA-II rank and crowding tiebreak.

    Solutions are accepted front-by-front in rank order.  When accepting an
    entire front would exceed the budget, the remaining slots are filled with
    the most-isolated (highest crowding distance) members of that front.

    Args:
        objectives: Objective matrix, shape ``(P, 3)``.  All values to
            minimise.
        n_elite: Number of elite solutions to select.

    Returns:
        Tuple of:
            - ``elite_indices``: Sorted list of ``n_elite`` row indices from
              *objectives* that form the elite set.
            - ``front_ranks``: Integer array of shape ``(P,)`` giving the
              Pareto rank (1-indexed) of every solution.
    """
    P = len(objectives)
    n_elite = min(n_elite, P)

    fronts = fast_non_dominated_sort(objectives)

    # Assign per-solution ranks (1-based)
    front_ranks = np.zeros(P, dtype=int)
    for rank, front in enumerate(fronts, start=1):
        for idx in front:
            front_ranks[idx] = rank

    elite: List[int] = []

    for front in fronts:
        if len(elite) + len(front) <= n_elite:
            elite.extend(front)
        else:
            # Fill remaining slots by crowding distance (descending)
            remaining = n_elite - len(elite)
            cd = crowding_distance(front, objectives)
            sorted_front = [front[i] for i in np.argsort(-cd)[:remaining]]
            elite.extend(sorted_front)
            break

        if len(elite) >= n_elite:
            break

    return sorted(elite), front_ranks

"""
Population manager tracking feasibility, calculating diversity contribution, and selecting individuals.
"""

from typing import List

from .individual import Individual


class Population:
    """
    Maintains feasible and infeasible subpopulations for HGS-ADC.
    """

    def __init__(self, target_size: int, nb_close: int):
        self.feas: List[Individual] = []
        self.inf: List[Individual] = []
        self.target_size = target_size
        self.nb_close = nb_close

    def add_individual(self, ind: Individual) -> None:
        """Adds an individual to the correct subpopulation."""
        if ind.is_feasible:
            self.feas.append(ind)
        else:
            self.inf.append(ind)

    def compute_diversity(self, subpop: List[Individual], T: int) -> None:
        """
        Calculates DC for every individual in the subpopulation.
        Metric: Hamming Pattern Distance + Route Broken Pairs Distance.
        """
        pop_size = len(subpop)
        if pop_size <= 1:
            for ind in subpop:
                ind.dc = 1.0
            return

        for i in range(pop_size):
            distances = []
            for j in range(pop_size):
                if i == j:
                    continue
                d = self._distance(subpop[i], subpop[j], T)
                distances.append(d)

            distances.sort()
            neighbors = min(self.nb_close, len(distances))
            if neighbors > 0:
                subpop[i].dc = sum(distances[:neighbors]) / neighbors
            else:
                subpop[i].dc = 0.0

    def _distance(self, ind_a: Individual, ind_b: Individual, T: int) -> float:
        """
        Calculate distance between two individuals.
        Hamming distance for patterns, broken pairs for routes.
        """
        # Pattern Hamming distance
        hamming = 0
        for p1, p2 in zip(ind_a.patterns, ind_b.patterns, strict=False):
            # Python 3.9 compatible bit count
            hamming += bin(int(p1) ^ int(p2)).count("1")

        # Broken pairs for routes
        broken_pairs = 0
        for t in range(T):
            if t < len(ind_a.giant_tours) and t < len(ind_b.giant_tours):
                ta = ind_a.giant_tours[t]
                tb = ind_b.giant_tours[t]

                # count adjacent pairs in ta missing in tb (undirected or directed)
                if len(ta) > 0 and len(tb) > 0:
                    set_b = set(zip(tb[:-1], tb[1:], strict=False))
                    for idx in range(len(ta) - 1):
                        if (ta[idx], ta[idx + 1]) not in set_b:
                            broken_pairs += 1

        return float(hamming + broken_pairs)

    def rank_and_survive(self, subpop: List[Individual], T: int) -> List[Individual]:
        """
        Ranks a subpopulation by Biased Fitness and culls to targeted size.
        """
        if len(subpop) <= self.target_size:
            return subpop

        # Calculate Fit rank (minimize fit -> rank 1 is best)
        subpop.sort(key=lambda x: x.fit)
        fit_ranks = {id(ind): idx + 1 for idx, ind in enumerate(subpop)}

        # Calculate DC
        self.compute_diversity(subpop, T)

        # Calculate DC rank (maximize DC -> rank 1 is best/most diverse)
        subpop_dc_sorted = sorted(subpop, key=lambda x: x.dc, reverse=True)
        dc_ranks = {id(ind): idx + 1 for idx, ind in enumerate(subpop_dc_sorted)}

        pop_size = len(subpop)
        nb_closer = min(self.nb_close, pop_size)

        for ind in subpop:
            f_rank = fit_ranks[id(ind)]
            d_rank = dc_ranks[id(ind)]
            ind.biased_fitness = f_rank + (1.0 - nb_closer / pop_size) * d_rank

        # Finally sort by biased_fitness (minimize)
        subpop.sort(key=lambda x: x.biased_fitness)

        # cull
        return subpop[: self.target_size]

    def trigger_survivor_selection(self, T: int) -> None:
        """Applies survival to both populations."""
        self.feas = self.rank_and_survive(self.feas, T)
        self.inf = self.rank_and_survive(self.inf, T)

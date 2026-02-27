"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.
"""

import random
from typing import List

import numpy as np

from .individual import Individual
from .split import LinearSplit


def ordered_crossover(p1: Individual, p2: Individual) -> Individual:
    """
    Apply Ordered Crossover (OX) to giant tours.
    """
    size = len(p1.giant_tour)
    a, b = sorted(random.sample(range(size), 2))

    child_gt = [0] * size
    child_gt[a : b + 1] = p1.giant_tour[a : b + 1]

    fill_pos = (b + 1) % size
    source_pos = (b + 1) % size

    p1_set = set(p1.giant_tour[a : b + 1])

    for _ in range(size):
        node = p2.giant_tour[source_pos]
        if node not in p1_set:
            child_gt[fill_pos] = node
            fill_pos = (fill_pos + 1) % size
        source_pos = (source_pos + 1) % size

    return Individual(child_gt)


def update_biased_fitness(population: List[Individual], nb_elite: int):
    """
    Update biased fitness based on profit rank and diversity rank.
    """
    if not population:
        return
    pop_size = len(population)
    if pop_size == 0:
        return

    # Rank by Profit
    population.sort(key=lambda x: x.profit_score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    # Diversity: Average distance to closest individuals
    for i, ind1 in enumerate(population):
        dists = []
        s1 = set(ind1.giant_tour)
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            s2 = set(ind2.giant_tour)
            # Hamming-like distance on set
            dist = len(s1.symmetric_difference(s2))
            dists.append(dist)
        dists.sort()
        # Avg of closest 5
        ind1.dist_to_parents = float(np.mean(dists[:5])) if dists else 0.0

    # Rank by Diversity
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Biased Fitness = Rank(Profit) + (1 - Elite/Size) * Rank(Diversity)
    factor = 1.0 - (float(nb_elite) / pop_size)
    for ind in population:
        ind.fitness = ind.rank_profit + factor * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit):
    """
    Decode giant tour and calculate metrics.
    """
    routes, profit = split_manager.split(ind.giant_tour)
    ind.routes = routes
    ind.profit_score = profit

    # Calculate cost/rev
    rev = 0.0
    cost = 0.0
    for r in routes:
        if not r:
            continue
        d = split_manager.dist_matrix[0, r[0]]
        rev += split_manager.wastes[r[0]] * split_manager.R
        for i in range(len(r) - 1):
            d += split_manager.dist_matrix[r[i], r[i + 1]]
            rev += split_manager.wastes[r[i + 1]] * split_manager.R
        d += split_manager.dist_matrix[r[-1], 0]
        cost += d * split_manager.C

    ind.cost = cost
    ind.revenue = rev

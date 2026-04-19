"""
Contextual Multi-Armed Bandit for Crossover Operator Selection.

This module implements contextual bandits (LinUCB, Thompson Sampling) for
dynamically selecting crossover operators based on population context.

Context Features:
    - Population diversity (measured by genetic distance)
    - Parent fitness similarity
    - Convergence rate (improvement velocity)
    - Iteration phase (early/mid/late)
    - Parent quality (elite vs non-elite)

Algorithms:
    1. **LinUCB (Linear Upper Confidence Bound)** - Li et al., 2010
       - Linear model with UCB exploration bonus
       - Efficient for contextual bandits

    2. **Thompson Sampling (Contextual)** - Agrawal & Goyal, 2013
       - Bayesian approach with posterior sampling
       - Excellent exploration-exploitation balance

    3. **ε-Greedy Contextual** - Baseline
       - Simple ε-greedy with context-aware rewards

Reference:
    Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.
    Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear Payoffs", ICML 2013.
"""

from collections import deque
from typing import Deque, List

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual


class ContextFeatureExtractor:
    """
    Extracts context features from parent individuals and population state.
    """

    def __init__(self, diversity_history_size: int = 10, improvement_history_size: int = 10):
        """Initialize feature extractor."""
        self.diversity_history: Deque[float] = deque(maxlen=diversity_history_size)
        self.improvement_history: Deque[float] = deque(maxlen=improvement_history_size)

    def extract_features(
        self,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
        iteration: int,
        progress: float = 0.0,
    ) -> np.ndarray:
        """
        Extract context feature vector for crossover operator selection.

        Args:
            p1: First parent.
            p2: Second parent.
            population: Current population.
            iteration: Current iteration number.
            progress: Normalized progress in [0, 1] (e.g., time or iteration).

        Returns:
            Feature vector (normalized to [0, 1]).
        """
        features = []

        # Feature 1: Parent fitness similarity (0 = identical, 1 = very different)
        if population:
            fitness_range = max(ind.profit_score for ind in population) - min(ind.profit_score for ind in population)
            fitness_diff = abs(p1.profit_score - p2.profit_score) / fitness_range if fitness_range > 1e-06 else 0.0
        else:
            fitness_diff = 0.5
        features.append(min(1.0, fitness_diff))

        # Feature 2: Parent genetic distance (Hamming distance on giant tour)
        if len(p1.giant_tour) > 0 and len(p2.giant_tour) > 0:
            common_nodes = set(p1.giant_tour) & set(p2.giant_tour)
            total_nodes = len(set(p1.giant_tour) | set(p2.giant_tour))
            genetic_distance = 1.0 - len(common_nodes) / max(total_nodes, 1)
        else:
            genetic_distance = 0.5
        features.append(genetic_distance)

        # Feature 3: Population diversity (average distance between individuals)
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        features.append(diversity)

        # Feature 4: Iteration progress (0 = early, 1 = late)
        features.append(min(1.0, max(0.0, progress)))

        # Feature 5: Parent quality (average of parent fitness ranks)
        if population:
            sorted_pop = sorted(population, key=lambda x: x.profit_score, reverse=True)
            rank_p1 = next((i for i, ind in enumerate(sorted_pop) if ind is p1), len(sorted_pop))
            rank_p2 = next((i for i, ind in enumerate(sorted_pop) if ind is p2), len(sorted_pop))
            avg_rank = (rank_p1 + rank_p2) / (2 * max(len(sorted_pop), 1))
        else:
            avg_rank = 0.5
        features.append(avg_rank)

        # Feature 6: Convergence rate (recent improvement velocity)
        convergence_rate = np.mean(self.improvement_history) if self.improvement_history else 0.5
        features.append(float(min(1.0, convergence_rate)))

        # Feature 7: Parent diversity contribution
        p1_diversity = self._individual_diversity(p1, population)
        p2_diversity = self._individual_diversity(p2, population)
        avg_parent_diversity = (p1_diversity + p2_diversity) / 2.0
        features.append(avg_parent_diversity)

        # Feature 8: Bias term (always 1.0 for linear models)
        features.append(1.0)

        return np.array(features, dtype=np.float64)

    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate average genetic diversity in population."""
        if len(population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                s1 = set(population[i].giant_tour)
                s2 = set(population[j].giant_tour)
                distance = len(s1.symmetric_difference(s2)) / max(len(s1 | s2), 1)
                total_distance += distance
                count += 1

        return total_distance / max(count, 1)

    def _individual_diversity(self, ind: Individual, population: List[Individual]) -> float:
        """Calculate average distance of individual to population."""
        if len(population) <= 1:
            return 0.0

        total_distance = 0.0
        count = 0

        s1 = set(ind.giant_tour)
        for other in population:
            if other is not ind:
                s2 = set(other.giant_tour)
                distance = len(s1.symmetric_difference(s2)) / max(len(s1 | s2), 1)
                total_distance += distance
                count += 1

        return total_distance / max(count, 1)

    def update_improvement(self, improvement: float):
        """Update improvement history."""
        self.improvement_history.append(improvement)

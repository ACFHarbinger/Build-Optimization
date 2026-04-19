"""
Enhanced HGS Evolution with Contextual Multi-Armed Bandits.

This module extends the standard HGS evolution operators with intelligent
crossover operator selection using contextual bandits.

Features:
    - LinUCB-based operator selection (default)
    - Thompson Sampling alternative
    - ε-Greedy baseline
    - Context-aware reward calculation
    - Adaptive operator portfolio

Reference:
    Vidal et al., "A hybrid genetic algorithm for multidepot and periodic VRP", 2012.
    Li et al., "Contextual Bandits for Adaptive Operator Selection", WWW 2010.
"""

import random
from typing import List, Optional

import numpy as np

from logic.src.policies.helpers.operators.crossover_recombination import CROSSOVER_NAMES, CROSSOVER_OPERATORS
from logic.src.policies.helpers.reinforcement_learning.agents.bandits import (
    EpsilonGreedyBandit,
)
from logic.src.policies.helpers.reinforcement_learning.agents.contextual_bandits import (
    ContextualThompsonSamplingAgent,
    LinUCBAgent,
)
from logic.src.policies.helpers.reinforcement_learning.features.context import ContextFeatureExtractor
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
    Individual,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit


class CMABEvolution:
    """
    Contextual Multi-Armed Bandit based evolutionary operators.

    This class orchestrates the dynamic selection of crossover operators using
    contextual reinforcement learning. It extracts environment and population
    features to form a 'context' vector, which is used by the bandit algorithm
    to predict which operator will be most effective.
    """

    def __init__(
        self,
        split_manager: LinearSplit,
        bandit_algorithm: str = "linucb",  # 'linucb', 'thompson', 'epsilon_greedy'
        quality_weight: float = 0.5,
        improvement_weight: float = 0.6,
        diversity_weight: float = 0.2,
        novelty_weight: float = 1.0,
        reward_threshold: float = 1e-6,
        default_reward: float = 5.0,
        rng: Optional[random.Random] = None,
        **kwargs,
    ):
        """
        Initialize the CMAB evolution manager.

        Args:
            split_manager: LinearSplit for evaluating individuals and splitting routes.
            bandit_algorithm: The RL algorithm to use (linucb, thompson, epsilon_greedy).
            quality_weight: Weight for the absolute quality of the offspring.
            improvement_weight: Weight for the improvement relative to parents.
            diversity_weight: Weight for the offspring's contribution to population diversity.
            novelty_weight: Weight for penalizing stagnant (non-novel) solutions.
            reward_threshold: Minimal improvement value to consider.
            default_reward: Baseline reward multiplier.
            rng: Random number generator instance.
            **kwargs: Additional parameters like alpha (for LinUCB) or lambda_prior.
        """
        self.split_manager = split_manager
        self.rng = rng if rng is not None else random.Random()
        # Seed for numpy generator used by new agents
        self.np_rng = np.random.default_rng()

        # Initialize feature extractor
        self.feature_extractor = ContextFeatureExtractor(
            diversity_history_size=kwargs.get("diversity_history_size", 10),
            improvement_history_size=kwargs.get("improvement_history_size", 10),
        )

        # Initialize bandit selector
        n_operators = len(CROSSOVER_OPERATORS)
        feature_dim = kwargs.get("feature_dim", 8)

        if bandit_algorithm == "linucb":
            self.bandit = LinUCBAgent(
                n_operators,
                feature_dim=feature_dim,
                alpha=kwargs.get("alpha", 1.0),
            )
        elif bandit_algorithm == "thompson":
            self.bandit = ContextualThompsonSamplingAgent(  # type: ignore[assignment]
                n_operators,
                feature_dim=feature_dim,
                lambda_prior=kwargs.get("lambda_prior", 1.0),
                noise_variance=kwargs.get("noise_variance", 0.1),
            )
        elif bandit_algorithm == "epsilon_greedy":
            self.bandit = EpsilonGreedyBandit(  # type: ignore[assignment]
                n_arms=n_operators,
                epsilon=kwargs.get("epsilon", 0.1),
                epsilon_decay=kwargs.get("epsilon_decay", 0.999),
                epsilon_min=kwargs.get("epsilon_min", 0.01),
            )
        else:
            raise ValueError(f"Unknown bandit algorithm: {bandit_algorithm}")

        self.bandit_algorithm = bandit_algorithm
        self.quality_weight = quality_weight
        self.improvement_weight = improvement_weight
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.reward_threshold = float(reward_threshold)
        self.default_reward = default_reward

        # Tracking
        self.iteration = 0

    def crossover(
        self,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
        iteration: int,
        progress: float = 0.0,
    ) -> Individual:
        """
        Execute crossover by selecting the optimal operator via bandit.

        Args:
            p1 (Individual): First parent.
            p2 (Individual): Second parent.
            population (List[Individual]): Current population for context extraction.
            iteration (int): Current global iteration.
            progress (float): Normalized search progress in [0, 1].

        Returns:
            Individual: The resulting offspring.
        """
        self.iteration = iteration

        # Step 1: Extract context features (geographical, diversity, progress)
        context = self.feature_extractor.extract_features(p1, p2, population, iteration, progress)

        # Step 2: Select action (crossover operator) using the bandit's upper confidence bound or sampling
        op_idx = self.bandit.select_action(context, self.np_rng)
        operator_name = CROSSOVER_NAMES[op_idx]
        operator_func = CROSSOVER_OPERATORS[operator_name]

        # Step 3: Apply the physical crossover operation
        child = operator_func(p1, p2, self.rng)  # type: ignore[operator]

        # Step 4: Map evaluation (SPLIT)
        self.evaluate(child)

        # Step 5: Reward Calculation (Multi-objective feedback)
        reward = self._calculate_reward(child, p1, p2, population)

        # Step 6: Update bandit with the outcome
        # If LinUCB: A = A + x*x.T, b = b + r*x
        if isinstance(self.bandit, (LinUCBAgent, ContextualThompsonSamplingAgent)):
            self.bandit.update(context, op_idx, reward)
        else:
            self.bandit.update(None, op_idx, reward, None, False)

        # Store metadata for performance tracking
        child.crossover_operator = operator_name
        child.crossover_reward = reward

        return child

    def _calculate_reward(
        self,
        child: Individual,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
    ) -> float:
        """
        Calculate a multi-objective reward signal for the crossover performance.

        The reward combines quality improvement, contribution to global progress,
        novelty, and population diversity to guide the bandit towards effective
        operator selection.

        Args:
            child (Individual): The newly created offspring.
            p1 (Individual): First parent.
            p2 (Individual): Second parent.
            population (List[Individual]): Current population for diversity measurement.

        Returns:
            float: A composite reward value scaled for the bandit update.
        """
        if not population:
            return self.default_reward

        # 1. Component: Elite Improvement (The 'Leapfrog' effect)
        # Reward based on how much the child improves over the BETTER parent.
        best_parent_profit = max(p1.profit_score, p2.profit_score)

        # Normalized improvement relative to parents
        improvement = (
            (child.profit_score - best_parent_profit) / abs(best_parent_profit)
            if abs(best_parent_profit) > self.reward_threshold
            else 0.0
        )

        # Scale improvement non-linearly to emphasize "Elite" breakthroughs
        improvement_reward = np.sign(improvement) * np.log1p(abs(improvement * 100))

        # 2. Component: Global Progress
        # Reward based on the child's rank relative to the entire population.
        global_best = max(ind.profit_score for ind in population)
        global_worst = min(ind.profit_score for ind in population)

        profit_range = global_best - global_worst
        quality_reward = (
            (child.profit_score - global_worst) / profit_range
            if profit_range > self.reward_threshold
            else self.default_reward / 10
        )

        # 3. Component: Diversity Credit
        # Crucial for preventing premature convergence in genetic search.
        child_diversity = self.feature_extractor._individual_diversity(child, population)

        # 4. Component: Penalty for Stagnation
        # Discourage operators that simply replicate parent genotypes.
        novelty_penalty = 0.0
        if child.profit_score == p1.profit_score or child.profit_score == p2.profit_score:
            novelty_penalty = -self.default_reward / 10

        # Weighted Sum of Rewards
        # We increase the weight of improvement_reward as it's the hardest to achieve
        total_reward = (
            (self.quality_weight * quality_reward)
            + (self.improvement_weight * improvement_reward)
            + (self.diversity_weight * child_diversity)
            + (self.novelty_weight * novelty_penalty)
        )

        # 5. Output: Tanh Soft-Clipping to [0, 10]
        # This ensures that even massive improvements don't cause Q-value explosion
        # while keeping the gradient sensitive near the 5.0 (neutral) mark.
        return self.default_reward * (np.tanh(total_reward) + 1.0)

    def evaluate(self, ind: Individual):
        """
        Evaluate individual using LinearSplit.

        Args:
            ind: Individual to evaluate.
        """
        routes, profit = self.split_manager.split(ind.giant_tour)
        ind.routes = routes
        ind.profit_score = profit

        # Calculate cost/revenue
        rev = 0.0
        cost = 0.0
        for r in routes:
            if not r:
                continue
            d = self.split_manager.dist_matrix[0, r[0]]
            rev += self.split_manager.wastes[r[0]] * self.split_manager.R
            for i in range(len(r) - 1):
                d += self.split_manager.dist_matrix[r[i], r[i + 1]]
                rev += self.split_manager.wastes[r[i + 1]] * self.split_manager.R
            d += self.split_manager.dist_matrix[r[-1], 0]
            cost += d * self.split_manager.C

        ind.cost = cost
        ind.revenue = rev

    def update_improvement(self, improvement: float):
        """
        Update feature extractor with improvement info.

        Args:
            improvement: Population improvement rate.
        """
        self.feature_extractor.update_improvement(improvement)

    def decay_exploration(self):
        """Decay exploration rate (for ε-greedy)."""
        if hasattr(self.bandit, "decay_epsilon"):
            self.bandit.decay_epsilon()

    def get_statistics(self) -> dict:
        """
        Get CMAB statistics for visualization.

        Returns:
            Dictionary of statistics.
        """
        stats = self.bandit.get_statistics()
        stats["bandit_algorithm"] = self.bandit_algorithm
        stats["iteration"] = self.iteration

        # Add operator names
        stats["operator_names"] = CROSSOVER_NAMES

        return stats


def update_biased_fitness(
    population: List[Individual],
    nb_elite: int,
    neighbor_size: int = 15,
):
    """
    Update biased fitness based on profit rank and diversity rank.
    Uses parameterless diversity weighting (Vidal 2022).

    BF(I) = Rank_P(I) + (1 - nb_elite / pop_size) * Rank_D(I)

    Args:
        population: List of individuals to update.
        nb_elite: Number of elite individuals to protect.
        neighbor_size: Number of nearest neighbors to consider.
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

    # Diversity: Average distance to closest individuals based on VISITED node sets
    for i, ind1 in enumerate(population):
        dists = []
        s1 = set(n for r in ind1.routes for n in r)
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            s2 = set(n for r in ind2.routes for n in r)
            # Hamming-like distance on set of visited nodes
            dist = len(s1.symmetric_difference(s2))
            dists.append(dist)
        dists.sort()
        # Avg of closest neighbor_size
        ind1.dist_to_parents = float(sum(dists[:neighbor_size]) / neighbor_size) if dists else 0.0

    # Rank by Diversity
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Biased Fitness = Rank(Profit) + diversity_weight * Rank(Diversity)
    # Parameterless Biased Fitness (Vidal 2022)
    diversity_weight = 1.0 - (nb_elite / pop_size)
    for ind in population:
        ind.fitness = ind.rank_profit + diversity_weight * ind.rank_diversity

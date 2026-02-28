"""
Soccer League Competition (SLC) algorithm for Build Optimization.

Models a population of builds (players) organised into teams.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import SLCParams


class SLCSolver(PolicyVizMixin):
    """
    Soccer League Competition solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: SLCParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the SLC algorithm.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()
        teams, stagnation, team_best_scores, best_build, best_score = self._init_league()

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            self._compete_intra_team(teams)
            self._compete_inter_team(teams)
            self._check_stagnation(teams, stagnation, team_best_scores)

            # Update superstar
            iter_best_build, iter_best_score = self._league_best(teams)
            if iter_best_score > best_score:
                best_build = iter_best_build.copy()
                best_score = iter_best_score

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                n_teams=self.params.n_teams,
            )

        return best_build, best_score

    def _init_league(self) -> Tuple[List[List[Tuple[np.ndarray, float]]], List[int], List[float], np.ndarray, float]:
        """Initialise teams, stagnation counters, and best scores."""
        teams: List[List[Tuple[np.ndarray, float]]] = []
        for _ in range(self.params.n_teams):
            team = []
            for _ in range(self.params.team_size):
                b = self.problem.random_solution()
                team.append((b, self.problem.evaluate(b)))
            teams.append(team)

        stagnation: List[int] = [0] * self.params.n_teams
        team_best_scores: List[float] = [max(p for _, p in team) for team in teams]
        best_build, best_score = self._league_best(teams)

        return teams, stagnation, team_best_scores, best_build, best_score

    def _compete_intra_team(self, teams: List[List[Tuple[np.ndarray, float]]]):
        """Intra-team competition: local perturbation."""
        for team in teams:
            for p_idx in range(len(team)):
                build, score = team[p_idx]
                new_build = self._perturb(build)
                new_score = self.problem.evaluate(new_build)
                if new_score > score:
                    team[p_idx] = (new_build, new_score)

    def _compete_inter_team(self, teams: List[List[Tuple[np.ndarray, float]]]):
        """Inter-team competition: probabilistic match."""
        team_indices = list(range(self.params.n_teams))
        random.shuffle(team_indices)
        for k in range(0, len(team_indices) - 1, 2):
            a_idx, b_idx = team_indices[k], team_indices[k + 1]
            fit_a = sum(p for _, p in teams[a_idx])
            fit_b = sum(p for _, p in teams[b_idx])

            p_win_a = 0.5
            if abs(fit_a) + abs(fit_b) > 0:
                p_win_a = max(0.1, min(0.9, fit_a / (fit_a + fit_b + 1e-9)))

            winner_idx = a_idx if random.random() < p_win_a else b_idx
            loser_idx = b_idx if winner_idx == a_idx else a_idx

            # Weakest player in losing team adopts structure from winner's best
            winner_best_build = max(teams[winner_idx], key=lambda x: x[1])[0]
            loser_worst_p_idx = int(np.argmin([p for _, p in teams[loser_idx]]))

            child = self._recombine(teams[loser_idx][loser_worst_p_idx][0], winner_best_build)
            child_score = self.problem.evaluate(child)
            teams[loser_idx][loser_worst_p_idx] = (child, child_score)

    def _check_stagnation(
        self, teams: List[List[Tuple[np.ndarray, float]]], stagnation: List[int], team_best_scores: List[float]
    ):
        """Stagnation check and team regeneration."""
        for t_idx, team in enumerate(teams):
            current_best = max(p for _, p in team)
            if current_best > team_best_scores[t_idx] + 1e-6:
                team_best_scores[t_idx] = current_best
                stagnation[t_idx] = 0
            else:
                stagnation[t_idx] += 1
                if stagnation[t_idx] >= self.params.stagnation_limit:
                    # Regenerate team
                    new_t = []
                    for _ in range(self.params.team_size):
                        b = self.problem.random_solution()
                        new_t.append((b, self.problem.evaluate(b)))
                    teams[t_idx] = new_t
                    stagnation[t_idx] = 0
                    team_best_scores[t_idx] = max(p for _, p in teams[t_idx])

    def _perturb(self, build: np.ndarray) -> np.ndarray:
        """Intra-team perturbation: small destroy and repair."""
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _recombine(self, loser_build: np.ndarray, winner_build: np.ndarray) -> np.ndarray:
        """Recombine loser's build with winner's best build."""
        new_build = loser_build.copy()
        # Adopt some slots from the winner
        mask = np.random.random(size=new_build.shape) < 0.3
        new_build[mask] = winner_build[mask]

        # Ensure feasibility
        if not self.problem.is_feasible(new_build):
            # Repair via destroy-repair
            partial = random_removal(new_build, self.params.n_removal, self.problem)
            new_build = greedy_insertion(partial, self.budget, self.problem)

        return new_build

    def _league_best(self, teams: List[List[Tuple[np.ndarray, float]]]) -> Tuple[np.ndarray, float]:
        """Return the best (build, score) across all teams."""
        best_s = -float("inf")
        best_b = np.array([])
        for team in teams:
            for build, score in team:
                if score > best_s:
                    best_s = score
                    best_b = build
        return best_b.copy(), best_s

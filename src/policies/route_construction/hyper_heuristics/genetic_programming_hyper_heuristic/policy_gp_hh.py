"""
GPHH Policy Adapter.

Adapts the Genetic Programming Hyper-Heuristic (GPHH) constructive heuristic
generator to the agnostic BaseRoutingPolicy interface.

Generates synthetic training environments (random Euclidean distance matrices)
to provide true spatial-topology diversity during GP tree evolution.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gp_hh import GPHHConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.params import GPHHParams
from logic.src.policies.route_construction.hyper_heuristics.genetic_programming_hyper_heuristic.solver import GPHHSolver

# ---------------------------------------------------------------------------
# Synthetic training environment factory
# ---------------------------------------------------------------------------


def _make_synthetic_training_envs(
    n_nodes: int,
    n_envs: int,
    capacity: float,
    R: float,
    C: float,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, Dict[int, float], List[int]]]:
    """
    Generate ``n_envs`` random Euclidean VRPP instances for GP training.

    Each environment has the same number of nodes as the test instance but
    with independently randomised 2-D coordinates and waste values, giving
    distinct spatial topologies.  This prevents the GP tree from overfitting
    to the test instance's geometry during evolution.

    Distance matrices are Euclidean (L2), scaled to the unit square.
    Waste values are drawn uniformly from (0, max_capacity / 4].

    Args:
        n_nodes: Number of customer nodes (excluding depot).
        n_envs: Number of training environments to generate.
        capacity: Vehicle capacity (for waste sampling).
        R: Revenue coefficient.
        C: Cost coefficient (unused here, kept for signature symmetry).
        rng: NumPy random generator for reproducibility.

    Returns:
        List of (dist_matrix, wastes, mandatory_nodes) triples.
    """
    envs: List[Tuple[np.ndarray, Dict[int, float], List[int]]] = []
    max_waste = max(capacity / 4.0, 1.0)

    for _ in range(n_envs):
        # Generate random 2-D coordinates for depot + n_nodes customers
        coords = rng.random((n_nodes + 1, 2)) * 100.0
        dm = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

        wastes = {i: float(rng.uniform(0.1, max_waste)) for i in range(1, n_nodes + 1)}
        envs.append((dm, wastes, []))  # No mandatory nodes in synthetic envs

    return envs


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("gphh")
class GPHHPolicy(BaseRoutingPolicy):
    """
    Genetic Programming Hyper-Heuristic (GP-HH) policy class.

    Evolves a GP tree that scores candidate insertions during constructive
    solution building, generating a learned construction heuristic.
    Provides synthetic training environments with diverse spatial topologies
    to promote generalizable GP trees.
    """

    def __init__(self, config: Optional[Union[GPHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GPHHConfig

    def _get_config_key(self) -> str:
        return "gp_hh"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the Genetic Programming Hyper-Heuristic (GPHH) solver logic.

        GPHH evolves a population of expression trees (learned construction
        heuristics) that decide node insertion priority. These trees are
        evaluated across multiple synthetic environments to ensure structural
        generalization across different spatial topologies.

        The best-evolved construction heuristic is then applied to the current
        problem state to produce the final routing plan.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                hyperparameters like `gp_pop_size`, `max_gp_generations`, etc.
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        seed = values.get("seed", 42)
        n_envs = int(values.get("n_training_instances", 3))
        n_nodes = len(sub_dist_matrix) - 1

        params = GPHHParams(
            gp_pop_size=int(values.get("gp_pop_size", 20)),
            max_gp_generations=int(values.get("max_gp_generations", 30)),
            tree_depth=int(values.get("tree_depth", 3)),
            tournament_size=int(values.get("tournament_size", 3)),
            time_limit=float(values.get("time_limit", 60.0)),
            parsimony_coefficient=float(values.get("parsimony_coefficient", 0.0)),
            candidate_list_size=int(values.get("candidate_list_size", 10)),
            n_training_instances=n_envs,
            training_sample_ratio=float(values.get("training_sample_ratio", 0.5)),
            seed=seed,
            vrpp=values.get("vrpp", True),
        )

        # Generate synthetic training environments with diverse spatial topologies
        rng = np.random.default_rng(seed)
        training_envs = _make_synthetic_training_envs(
            n_nodes=n_nodes,
            n_envs=n_envs,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            rng=rng,
        )

        solver = GPHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            training_environments=training_envs,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost

"""
Mixed-Integer Programming Assignment Module for Fisher & Jaikumar Clustering.

This module implements an exact optimization approach for assigning nodes to seed
clusters using Gurobi's MIP solver. The formulation maximizes total profit
(revenue - cost) subject to capacity and assignment constraints.

This is an extension of the original Fisher & Jaikumar algorithm that uses
exact optimization instead of greedy heuristics for the assignment phase.
Specifically, we introduce a Prize-Collecting adaptation of the GAP
objective to handle profit-aware vehicle routing (VRPP).

Reference:
    Fisher, M. L., & Jaikumar, R. (1981). "A generalized assignment heuristic
    for vehicle routing". Networks, 11(2), 109-124.
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017). "A Variant Fisher
    and Jaikumar Algorithm to Solve Capacitated Vehicle Routing Problem".
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def _add_variables_and_objective(
    model: "gp.Model",
    mandatory: List[int],
    seeds: List[int],
    costs: Dict[Tuple[int, int], float],
    wastes: Dict[int, float],
    R: float,
    C: float,
    objective: str,
) -> Dict[Tuple[int, int], "gp.Var"]:
    """Helper to add variables and setting objective for the MIP model."""
    x = {}
    if objective == "minimize_cost":
        model.ModelSense = GRB.MINIMIZE
        for i in mandatory:
            for k_idx in range(len(seeds)):
                x[i, k_idx] = model.addVar(vtype=GRB.BINARY, obj=costs[i, k_idx], name=f"x_{i}_{k_idx}")
    else:  # objective == "maximize_profit"
        model.ModelSense = GRB.MAXIMIZE
        for i in mandatory:
            for k_idx in range(len(seeds)):
                profit = (wastes.get(i, 0.0) * R) - (costs[i, k_idx] * C)
                x[i, k_idx] = model.addVar(vtype=GRB.BINARY, obj=profit, name=f"x_{i}_{k_idx}")
    return x


def assign_exact_mip(
    seeds: List[int],
    mandatory: List[int],
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    distance_matrix: np.ndarray,
    time_limit: float = 60.0,
    objective: str = "minimize_cost",
) -> Optional[List[List[int]]]:
    """
    Assign nodes to seeds by solving the Generalized Assignment Problem (GAP)
    as an exact Mixed-Integer Program using Gurobi.

    Ref: Fisher & Jaikumar (1981), Sultana & Akhand (2017).

    IMPORTANT: The 'maximize_profit' objective is a novel metaheuristic
    extension for Prize-Collecting VRPs (VRPP). It is mathematically
    distinct from the original GAP formulation presented in Sultana &
    Akhand (2017), which focuses solely on cost minimization.

    Mathematical Formulation (GAP):
        Minimize: Σ_i Σ_k (c_ik * x_ik)
        Subject to:
            Σ_k x_ik <= 1                    ∀i ∈ mandatory  (assignment)
            Σ_i w_i * x_ik <= capacity       ∀k ∈ seeds    (capacity)
            x_kk = 1                         ∀k ∈ seeds    (seed constraint)
            x_ik ∈ {0, 1}                    ∀i, k

        Where:
            c_ik = D_Si + D_i0 - D_S (S=seed, i=node, 0=depot)

            Note: We explicitly enforce c_kk = 0. This is a mathematical necessity
            to isolate the marginal cost of node i's insertion into cluster k.
            Since node k is the seed of its own cluster, its "insertion cost"
            relative to itself is zero by definition, ensuring it remains the
            anchor of that cluster without solver-induced artifacts.
            R = revenue per unit of waste
            C = cost per unit of distance

    Args:
        seeds: List of seed node indices (one per initial cluster).
        mandatory: List of all nodes that need to be assigned.
        wastes: Dictionary mapping node indices to waste quantities.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit of waste (used only in maximize_profit mode).
        C: Cost per unit of distance (used only in maximize_profit mode).
        distance_matrix: Pre-computed all-pairs distance matrix.
        time_limit: Maximum time in seconds for optimization (default: 60.0).
            The actual time used is 20% of this value.
        objective: Objective function ("minimize_cost" or "maximize_profit").
            Use "minimize_cost" for benchmark comparisons against A-VRP dataset.

    Returns:
        List of clusters, where each cluster is a list of node indices.
        Returns None if Gurobi is unavailable or MIP fails.

    Raises:
        ValueError: If objective is not "minimize_cost" or "maximize_profit".

    Note:
        - Requires gurobipy with valid license.
        - If Gurobi is unavailable or MIP fails, returns None to signal fallback needed.
        - Allocates 20% of total time_limit to the clustering phase.
        - For benchmark compliance with A-VRP dataset, use objective="minimize_cost".

    Example (Benchmark Mode):
        >>> seeds = [5, 12, 20]
        >>> mandatory = list(range(1, 21))
        >>> wastes = {i: 10.0 for i in mandatory}
        >>> capacity = 100.0
        >>> R, C = 5.0, 1.0
        >>> distance_matrix = np.random.rand(21, 21)
        >>> # Minimize cost for benchmark comparison
        >>> clusters = assign_exact_mip(seeds, mandatory, wastes, capacity, R, C,
        ...                             distance_matrix, objective="minimize_cost")

    Example (Simulation Mode):
        >>> # Maximize profit for simulation
        >>> clusters = assign_exact_mip(seeds, mandatory, wastes, capacity, R, C,
        ...                             distance_matrix, objective="maximize_profit")
    """
    # Validate objective parameter
    if objective not in ["minimize_cost", "maximize_profit"]:
        raise ValueError(
            f"Invalid objective: '{objective}'. "
            f"Must be 'minimize_cost' (benchmark mode) or 'maximize_profit' (simulation mode)."
        )

    # Check if Gurobi is available
    if not GUROBI_AVAILABLE:
        return None

    try:
        clusters: List[List[int]] = [[] for _ in range(len(seeds))]

        # 1. Precompute insertion costs: c_ik = D_Si + D_i0 - D_S
        costs = {}
        for i in mandatory:
            for k_idx, k in enumerate(seeds):
                if i == k:
                    # Seed insertion cost into its own cluster is exactly zero.
                    # This prevents the solver from trying to 'optimize' the seed's
                    # position relative to itself and ensures x_kk = 1 is always
                    # the cheapest assignment for the seed.
                    costs[i, k_idx] = 0.0
                else:
                    costs[i, k_idx] = distance_matrix[k, i] + distance_matrix[i, 0] - distance_matrix[k, 0]

        # 2. Setup Gurobi Environment and Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)  # Silence solver output
        env.start()
        model = gp.Model("Fisher_Jaikumar_GAP", env=env)

        # 3. Decision Variables and Objective Function
        x = _add_variables_and_objective(model, mandatory, seeds, costs, wastes, R, C, objective)

        # 4. Constraint: Assignment (Each node to AT MOST one cluster)
        for i in mandatory:
            model.addConstr(gp.quicksum(x[i, k_idx] for k_idx in range(len(seeds))) <= 1, name=f"assign_{i}")

        # 5. Constraint: Vehicle Capacity
        for k_idx in range(len(seeds)):
            model.addConstr(
                gp.quicksum(wastes.get(i, 0.0) * x[i, k_idx] for i in mandatory) <= capacity, name=f"cap_{k_idx}"
            )

        # 6. Constraint: Seeds must be assigned to their own clusters
        for k_idx, k in enumerate(seeds):
            if k in mandatory:
                model.addConstr(x[k, k_idx] == 1, name=f"seed_{k}")

        # 7. Solve the GAP
        # Allocate 20% of the total time budget to the exact clustering phase
        gap_time_limit = max(1.0, time_limit * 0.20)
        model.setParam("TimeLimit", gap_time_limit)
        model.optimize()

        # 8. Extract Solution
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            for i in mandatory:
                for k_idx in range(len(seeds)):
                    if x[i, k_idx].X > 0.5:
                        clusters[k_idx].append(i)
                        break
        else:
            # MIP is infeasible or failed - signal fallback needed
            return None

        return clusters

    except Exception:
        # If Gurobi fails for any reason, signal fallback needed
        return None

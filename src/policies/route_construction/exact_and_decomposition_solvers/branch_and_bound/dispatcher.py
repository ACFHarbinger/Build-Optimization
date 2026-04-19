"""
Branch-and-Bound Optimizer Interface.

Provides a unified dispatcher for selecting between different B&B formulations:
- MTZ (Miller-Tucker-Zemlin): Compact formulation with load variables
- DFJ (Dantzig-Fulkerson-Johnson): Lazy subtour elimination with Gurobi's B&B

This architecture mirrors the SWC-TCF dispatcher pattern, allowing seamless
switching between mathematical formulations based on problem characteristics.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .dfj import run_bb_dfj
from .lr_uop import run_bb_lr_uop
from .mtz import run_bb_mtz
from .params import BBParams


def run_bb_optimizer(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: Optional[BBParams] = None,
    mandatory_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
    **kwargs: Any,
) -> Tuple[List[List[int]], float]:
    """
    Solve VRPP using the specified Branch-and-Bound formulation.

    This dispatcher routes the problem to the appropriate B&B implementation
    based on the formulation parameter. The choice of formulation affects:
    - Subtour elimination mechanism
    - LP relaxation tightness
    - Computational performance
    - Memory footprint

    **Formulation Comparison**:

    **MTZ (Miller-Tucker-Zemlin)**:
    - Compact formulation with O(n²) constraints.
    - Load variables track accumulated waste.
    - Subtours prevented implicitly in LP relaxation.
    - **CRITICAL WARNING**: While compact, the MTZ LP relaxation is notoriously
      weak. It often yields highly fractional lower bounds far from the integer
      optimal, leading to exponentially larger B&B trees compared to stronger
      formulations.
    - Custom B&B tree management (Python-based) allows for full state observability.
    - Better for: Small instances, research into neural branching heuristics.

    **DFJ (Dantzig-Fulkerson-Johnson)**:
    - Exponential constraints added lazily as "cuts".
    - Delegates B&B to Gurobi's internal high-performance C-engine.
    - Subtours eliminated dynamically via callbacks.
    - **ADVANTAGE**: Provides a significantly tighter LP relaxation because the
      lazy subtour elimination constraints effectively cut off fractional cycles,
      leading to faster convergence.
    - Better for: Large production-scale instances.

    **Research Disclaimer**:
    Evaluating learned branching heuristics on a weak MTZ formulation does not
    guarantee identical performance or ranking on stronger formulations (like DFJ).
    This discrepancy should be explicitly addressed in academic disclosures.

    Args:
        dist_matrix: Symmetric distance matrix (n x n).
        wastes: Mapping of customer IDs to fill levels/profits.
        capacity: Vehicle payload capacity.
        R: Revenue coefficient per unit collected.
        C: Cost coefficient per unit distance.
        params: Standardized BB parameters.
        mandatory_indices: Set of mandatory customer nodes.
        env: Optional Gurobi environment for resource management.
        recorder: Optional telemetry recorder for state tracking.

    Returns:
        Tuple of (routes, objective_value).

    Raises:
        ValueError: If formulation is not "mtz" or "dfj".

    Examples:
        >>> # Use DFJ for large instances
        >>> routes, obj = run_bb_optimizer(
        ...     dist_matrix, wastes, capacity, R, C, params=params
        ... )
    """
    if params is None:
        params = BBParams()

    formulation = params.formulation

    if formulation == "mtz":
        return run_bb_mtz(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_indices=mandatory_indices,
            env=env,
            recorder=recorder,
        )
    elif formulation == "dfj":
        return run_bb_dfj(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_indices=mandatory_indices,
            env=env,
            recorder=recorder,
        )
    elif formulation == "lr_uop":
        return run_bb_lr_uop(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params,
            mandatory_indices=mandatory_indices,
            env=env,
            recorder=recorder,
        )
    else:
        raise ValueError(
            f"Unknown B&B formulation: '{formulation}'. "
            f"Supported formulations: 'mtz', 'dfj', 'lr_uop'. "
            f"Use 'mtz' for compact MTZ formulation, 'dfj' for lazy DFJ cuts, "
            f"or 'lr_uop' for Lagrangian Relaxation with uncapacitated OP bounding."
        )

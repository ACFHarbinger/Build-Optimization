"""
HULK Hyper-Heuristic Package.

HULK: Hyper-heuristic Using unstringing/stringing with Local search and K-opt

A selection hyper-heuristic that adaptively chooses between:
- Unstringing operators (Type I-IV) for solution destruction
- Stringing operators (Type I-IV) for solution reconstruction
- Local search operators (2-opt, 3-opt, swap, relocate) for improvement

Reference:
    Müller, L. F., & Bonilha, I. (2022). "Hyper-Heuristic Based on ACO
    and Local Search for Dynamic Optimization Problems."
    Algorithms, 15(1), 9. https://doi.org/10.3390/a15010009
"""

from typing import Any, List, Optional, Tuple

from .hulk import HULKSolver
from .params import HULKParams

__all__ = ["HULKSolver", "HULKParams", "run_hulk"]


def run_hulk(context: Any, params: Optional[HULKParams] = None) -> Tuple[List[List[int]], float, float]:
    """
    Run HULK hyper-heuristic on the given problem context.

    Args:
        context: Problem context with dist_matrix, wastes, capacity, etc.
        params: HULK parameters (uses defaults if None).

    Returns:
        Tuple of (best_routes, best_profit, best_cost).
    """
    if params is None:
        params = HULKParams()

    solver = HULKSolver(
        dist_matrix=context.dist_matrix,
        wastes=context.wastes,
        capacity=context.capacity,
        R=context.R,
        C=context.C,
        params=params,
        mandatory_nodes=getattr(context, "mandatory_nodes", None),
    )

    return solver.solve()

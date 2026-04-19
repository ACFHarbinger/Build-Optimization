"""
Double-Bridge Move (Random 4-opt) Perturbation Module.

Implements the double-bridge move, a classical 4-opt perturbation that
slices a route into 4 segments by choosing 3 random cut points and
reconnects them in a non-sequential order to escape local optima.

Given segments A, B, C, D the reconnection is: A + C + B + D.
This creates a configuration that cannot be reached by any sequence of
2-opt or 3-opt moves, making it a powerful diversification tool.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.double_bridge import double_bridge
    >>> # Using LocalSearch
    >>> success = double_bridge(ls, r_idx=0, rng=rng)
    >>> # Using raw route plan
    >>> success = double_bridge(plan, r_idx=0, rng=rng)
"""

from random import Random
from typing import Any, Optional


def double_bridge(ls_or_plan: Any, r_idx: int, rng: Optional[Any] = None) -> bool:
    """
    Double-bridge (4-opt) perturbation on a single route.

    Supports both LocalSearch instances and raw route plans (List[List[int]]).
    Selects 3 random cut points, slicing the route into 4 segments
    A, B, C, D and reconnects as A + C + B + D.

    Args:
        ls_or_plan: LocalSearch instance or List[List[int]] route plan.
        r_idx:      Index of the route to perturb.
        rng:        Random number generator (Random or np.random.Generator).

    Returns:
        bool: True if the move was applied, False otherwise.
    """
    if rng is None:
        rng = Random()

    # 1. Determine context and extract routes
    is_ls = hasattr(ls_or_plan, "routes")
    routes = ls_or_plan.routes if is_ls else ls_or_plan
    route = routes[r_idx]
    n = len(route)

    if n < 4:
        return False

    # 2. Generate 3 sorted cut points
    # Logic handles both random.Random (LS) and numpy.Generator (LLH Pools)
    if hasattr(rng, "sample"):
        # random.Random
        cuts = sorted(rng.sample(range(1, n), 3))
    else:
        # numpy.random.Generator or numpy.random.  Use getattr to satisfy Mypy
        # which thinks rng must be random.Random from the earlier import.
        choice_func: Any = rng.choice
        choice_res = choice_func(n - 1, size=3, replace=False)
        # Shift results to [1, n)
        cuts = sorted([int(x) + 1 for x in choice_res])

    if len(cuts) < 3:
        return False

    c1, c2, c3 = cuts[0], cuts[1], cuts[2]

    # 3. Slice into 4 segments
    seg_a = route[:c1]
    seg_b = route[c1:c2]
    seg_c = route[c2:c3]
    seg_d = route[c3:]

    # 4. Reconnect: A + C + B + D
    # This specific 4-edge swap creates a bridge unreachable by k < 4 opt.
    routes[r_idx] = seg_a + seg_c + seg_b + seg_d

    if is_ls:
        ls_or_plan._update_map({r_idx})

    return True

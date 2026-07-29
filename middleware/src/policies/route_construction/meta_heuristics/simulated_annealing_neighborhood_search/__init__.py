"""
Public exports for look-ahead auxiliary route construction and optimization.

This package provides a comprehensive suite of utilities for solving profit-
based vehicle routing problems, including mutation operators, feasibility
checks, performance metrics, and high-level search orchestrators.
"""

from .common.routes import create_points as create_points
from .common.solution_initialization import (
    compute_initial_solution as compute_initial_solution,
)
from .common.update import (
    add_bins_to_collect as add_bins_to_collect,
)
from .common.update import (
    get_next_collection_day as get_next_collection_day,
)
from .common.update import (
    should_bin_be_collected as should_bin_be_collected,
)
from .common.update import (
    update_fill_levels_after_first_collection as update_fill_levels_after_first_collection,
)
from .heuristics.sans import (
    improved_simulated_annealing as improved_simulated_annealing,
)
from .refinement.route_search import (
    find_solutions as find_solutions,
)

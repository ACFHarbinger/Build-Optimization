"""
Crossover Operators for Hybrid Genetic Search.

This module implements seven advanced crossover operators for VRP genetic algorithms:

1. **Ordered Crossover (OX)** - Davis, 1985
   - Preserves relative order of elements
   - Good for TSP-like problems

2. **Random Node Inheritance Crossover** - Custom
   - Focuses on which nodes to inherit, not their positions
   - Excellent for node selection in VRPP

3. **Selective Route Exchange Crossover (SREX)** - Custom
   - Exchanges complete routes between parents
   - Preserves good route structures

4. **Generalized Partition Crossover (GPX)** - Whitley et al., 2009
   - Graph-based recombination using uncommon edges
   - Creates offspring from edge unions

5. **Edge Recombination Crossover (ERX)** - Whitley, 1989
   - Preserves edges from parents
   - Builds offspring by following edge adjacencies

6. **Capacity-Aware ERX (C-ERX)** - VRPP Adaptation
   - ERX with capacity constraints and profit tie-breaking

7. **Route-based Profit GPX (RP-GPX)** - VRPP Adaptation
   - GPX operating on decoded routes with profit packing

Reference:
    Vidal et al., "A hybrid genetic algorithm for multidepot and periodic VRP", 2012.
    Nagata & Bräysy, "Edge assembly-based memetic algorithm", 2009.
"""

from .edge_recombination import capacity_aware_erx, edge_recombination_crossover
from .generalized_partition import (
    generalized_partition_crossover,
    route_profit_gpx_crossover,
)
from .ordered import ordered_crossover
from .position_independent import position_independent_crossover
from .random_node_inheritance import random_node_inheritance_crossover
from .selective_route_exchange import selective_route_exchange_crossover

CROSSOVER_OPERATORS = {
    "OX": ordered_crossover,
    "RANDOM_NODE": random_node_inheritance_crossover,
    "PIX": position_independent_crossover,
    "SREX": selective_route_exchange_crossover,
    "GPX": generalized_partition_crossover,
    "ERX": edge_recombination_crossover,
    "C-ERX": capacity_aware_erx,
    "RP-GPX": route_profit_gpx_crossover,
}

CROSSOVER_NAMES = list(CROSSOVER_OPERATORS.keys())

__all__ = [
    "CROSSOVER_NAMES",
    "CROSSOVER_OPERATORS",
    "capacity_aware_erx",
    "edge_recombination_crossover",
    "generalized_partition_crossover",
    "ordered_crossover",
    "position_independent_crossover",
    "random_node_inheritance_crossover",
    "route_profit_gpx_crossover",
    "selective_route_exchange_crossover",
]

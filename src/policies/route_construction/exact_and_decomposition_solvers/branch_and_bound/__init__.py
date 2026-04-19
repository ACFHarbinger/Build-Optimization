"""
Branch-and-Bound Policy Module.

This package implements the foundational Branch-and-Bound (BB) algorithm for exact
combinatorial optimization, specifically specialized for the Vehicle Routing
Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP).

Implementation follows the methodology proposed by Land and Doig (1960), utilizing
Linear Programming (LP) relaxations to provide rigorous mathematical bounds during
tree exploration.

**Architecture**:
- `mtz.py`: MTZ (Miller-Tucker-Zemlin) compact formulation
- `dfj.py`: DFJ (Dantzig-Fulkerson-Johnson) lazy cut formulation
- `dispatcher.py`: Unified interface for formulation selection
- `policy_bb.py`: Policy adapter integrating with the simulation framework
"""

from .dfj import run_bb_dfj
from .dispatcher import run_bb_optimizer
from .lr_uop import run_bb_lr_uop
from .mtz import run_bb_mtz
from .policy_bb import BranchAndBoundPolicy

__all__ = [
    "BranchAndBoundPolicy",
    "run_bb_optimizer",
    "run_bb_mtz",
    "run_bb_dfj",
    "run_bb_lr_uop",
]

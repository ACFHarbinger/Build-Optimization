"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Adapter Package.
"""

from .dispatcher import run_swc_tcf_optimizer
from .gurobi import _run_gurobi_optimizer
from .policy_swc_tcf import SWCTCFPolicy

__all__ = [
    "_run_gurobi_optimizer",
    "run_swc_tcf_optimizer",
    "SWCTCFPolicy",
]

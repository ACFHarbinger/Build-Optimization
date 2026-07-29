"""
Exact Stochastic Dynamic Programming (SDP) Policy.

Implements Backward Induction to solve small SCWCVRP/SIRP instances optimally.
"""

from .esdp_engine import ExactSDPEngine
from .params import SDPParams

__all__ = [
    "SDPParams",
    "ExactSDPEngine",
]

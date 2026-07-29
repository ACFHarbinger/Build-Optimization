"""
Volleyball Premier League (VPL) algorithm.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043
"""

from .params import VPLParams
from .policy_vpl import VPLPolicy
from .solver import VPLSolver

__all__ = ["VPLParams", "VPLSolver", "VPLPolicy"]

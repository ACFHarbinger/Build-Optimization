"""
SISR Runner Module.

This module provides a convenience entry point for running the SISR solver
with a dictionary of configuration values.

Attributes:
    None

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal.sisr import run_sisr
    >>> result = run_sisr(dist_matrix, wastes, ...)
"""

from .params import SISRParams
from .solver import SISRSolver


def run_sisr(dist_matrix, wastes, capacity, R, C, values, **kwargs):
    """
    Convenience entry point for SISR.
    """
    params = SISRParams(
        time_limit=values.get("time_limit", 10.0),
        max_iterations=values.get("max_iterations", 1000),
        start_temp=values.get("start_temp", 100.0),
        cooling_rate=values.get("cooling_rate", 0.995),
        max_string_len=values.get("max_string_len", 10),
        avg_string_len=values.get("avg_string_len", 3.0),
        blink_rate=values.get("blink_rate", 0.01),
        destroy_ratio=values.get("destroy_ratio", 0.2),
    )
    solver = SISRSolver(dist_matrix, wastes, capacity, R, C, params)
    return solver.solve()

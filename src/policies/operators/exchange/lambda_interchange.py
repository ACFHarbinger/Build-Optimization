"""
Lambda-Interchange Operator Module.

This module implements a wrapper for the Lambda-Interchange neighborhood search,
which systematically explores cross-exchange moves between all pairs of routes
with segment lengths up to a specified maximum (lambda_max).

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.exchange.lambda_interchange import lambda_interchange
    >>> improved = lambda_interchange(ls, lambda_max=2)
"""

from typing import Any

from .cross import cross_exchange


def lambda_interchange(
    ls: Any,
    lambda_max: int = 2,
) -> bool:
    """
    λ-Interchange neighborhood search.

    Systematically explores cross-exchange moves with segments up to
    length λ. This is a wrapper that explores the full neighborhood.

    Args:
        ls: LocalSearch instance.
        lambda_max: Maximum segment length to exchange.

    Returns:
        bool: True if any improving move was found.
    """
    improved = False

    for r_a in range(len(ls.routes)):
        for r_b in range(r_a + 1, len(ls.routes)):
            route_a = ls.routes[r_a]
            route_b = ls.routes[r_b]

            for seg_a_len in range(lambda_max + 1):
                for seg_b_len in range(lambda_max + 1):
                    if seg_a_len == 0 and seg_b_len == 0:
                        continue

                    for seg_a_start in range(max(1, len(route_a) - seg_a_len + 1)):
                        for seg_b_start in range(max(1, len(route_b) - seg_b_len + 1)):
                            if cross_exchange(
                                ls,
                                r_a,
                                seg_a_start,
                                seg_a_len,
                                r_b,
                                seg_b_start,
                                seg_b_len,
                            ):
                                improved = True
                                # Restart search after improvement
                                return True

    return improved

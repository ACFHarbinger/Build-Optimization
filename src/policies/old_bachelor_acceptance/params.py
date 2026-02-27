"""
Configuration parameters for the Old Bachelor Acceptance (OBA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OBAParams:
    """
    Configuration for the OBA solver.

    OBA uses a non-monotonic oscillating threshold.  The threshold dilates
    (becomes more permissive) after consecutive rejections and contracts
    (becomes stricter) after consecutive acceptances.

    Attributes:
        dilation: Amount to widen threshold after a rejection.
        contraction: Amount to tighten threshold after an acceptance.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    dilation: float = 5.0
    contraction: float = 2.0
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0

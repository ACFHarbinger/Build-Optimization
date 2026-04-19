"""
SISR Parameters Module.

This module defines the configuration parameters for the
Slack Induction by String Removal (SISR) algorithm.

Attributes:
    None

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal.params import SISRParams
    >>> params = SISRParams(time_limit=10.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SISRParams:
    """
    Configuration parameters for the SISR solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of SISR iterations.
        start_temp: Initial temperature for simulated annealing.
        cooling_rate: Temperature decay factor per iteration.
        max_string_len: Maximum length of string to remove.
        avg_string_len: Average length of strings to remove.
        blink_rate: Probability of 'blinking' during insertion.
        destroy_ratio: Fraction of solution to destroy per iteration.
        vrpp: Whether to solve the VRPP variant.
        profit_aware_operators: Whether to use profit-aware operators.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 10.0
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> SISRParams:
        """Create SISRParams from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            SISRParams instance with values from config.
        """
        return cls(
            time_limit=getattr(config, "time_limit", 10.0),
            max_iterations=getattr(config, "max_iterations", 1000),
            start_temp=getattr(config, "start_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.995),
            max_string_len=getattr(config, "max_string_len", 10),
            avg_string_len=getattr(config, "avg_string_len", 3.0),
            blink_rate=getattr(config, "blink_rate", 0.01),
            destroy_ratio=getattr(config, "destroy_ratio", 0.2),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

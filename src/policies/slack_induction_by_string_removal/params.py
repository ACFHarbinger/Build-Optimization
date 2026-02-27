"""
SISR Parameters Module.

This module defines the configuration parameters for the Slack Induction by
String Removal (SISR) algorithm.

Attributes:
    None

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal.params import SISRParams
    >>> params = SISRParams(time_limit=10.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import SISRConfig


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
    """

    time_limit: float = 10.0
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2

    @classmethod
    def from_config(cls, config: SISRConfig) -> SISRParams:
        """Create SISRParams from a SISRConfig dataclass.

        Args:
            config: SISRConfig dataclass with solver parameters.

        Returns:
            SISRParams instance with values from config.
        """
        return cls(
            time_limit=config.time_limit,
            max_iterations=config.max_iterations,
            start_temp=config.start_temp,
            cooling_rate=config.cooling_rate,
            max_string_len=config.max_string_len,
            avg_string_len=config.avg_string_len,
            blink_rate=config.blink_rate,
            destroy_ratio=config.destroy_ratio,
        )

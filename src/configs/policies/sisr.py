"""
SISR (Slack Induction by String Removal) configuration.
"""

from dataclasses import dataclass


@dataclass
class SISRConfig:
    engine: str = "sisr"
    time_limit: float = 10.0
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2

"""
Decoding Strategy Config module.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DecodingConfig:
    """Configuration for decoding strategies."""

    strategy: str = "greedy"  # greedy, sampling, beam_search
    beam_width: Any = 1  # For beam search or sampling size
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tanh_clipping: float = 0.0
    mask_logits: bool = True
    multistart: bool = False
    num_starts: int = 1
    select_best: bool = False

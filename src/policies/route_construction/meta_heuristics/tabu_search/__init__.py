"""
Tabu Search (TS) module.

Implements classical Tabu Search metaheuristic with:
- Short-term memory (recency-based tabu list)
- Long-term memory (frequency-based memory)
- Aspiration criteria
- Intensification and diversification strategies
- Elite solution pool
- Path relinking

Based on Fred Glover's "Tabu Search Fundamentals and Uses" (1995).
"""

from .params import TSParams
from .policy_ts import TSPolicy
from .solver import TSSolver

__all__ = ["TSParams", "TSPolicy", "TSSolver"]

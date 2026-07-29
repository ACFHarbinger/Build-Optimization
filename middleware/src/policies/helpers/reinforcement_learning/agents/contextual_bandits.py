"""
Contextual Multi-Armed Bandit (CMAB) agents.

Provides linear and non-linear agents for operator selection and parameter
control based on environmental context.
"""

from .contextual.base import ContextualBanditAgent
from .contextual.gpcmab import GPCMABAgent
from .contextual.linucb import LinUCBAgent
from .contextual.thompson import ContextualThompsonSamplingAgent

__all__ = [
    "ContextualBanditAgent",
    "LinUCBAgent",
    "ContextualThompsonSamplingAgent",
    "GPCMABAgent",
]

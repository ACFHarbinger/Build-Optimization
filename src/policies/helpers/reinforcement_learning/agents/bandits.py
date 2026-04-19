"""
Multi-Armed Bandit (MAB) agents.

Provides classic multi-armed bandit algorithms for operator selection
without environmental context.
"""

from .bandits.base import BanditAgent
from .bandits.epsilon_greedy import EpsilonGreedyBandit
from .bandits.exp3 import EXP3Agent
from .bandits.softmax import SoftmaxBandit
from .bandits.thompson import ThompsonSamplingBandit
from .bandits.ucb import DiscountedUCBBandit, SlidingWindowUCBBandit, UCBBandit

__all__ = [
    "BanditAgent",
    "EpsilonGreedyBandit",
    "UCBBandit",
    "SoftmaxBandit",
    "ThompsonSamplingBandit",
    "DiscountedUCBBandit",
    "SlidingWindowUCBBandit",
    "EXP3Agent",
]

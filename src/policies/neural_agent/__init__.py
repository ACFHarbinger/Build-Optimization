"""
Neural Agent Package.

This package provides the NeuralAgent implementation, which serves as a wrapper
around deep reinforcement learning models for the WSmart-Route system.

Attributes:
    __all__ (List[str]): List of public objects exported by this module.

Example:
    >>> from logic.src.policies.neural_agent import NeuralAgent
    >>> agent = NeuralAgent(model)
    >>> action = agent.act(state)
"""

from .agent import NeuralAgent

__all__ = ["NeuralAgent"]

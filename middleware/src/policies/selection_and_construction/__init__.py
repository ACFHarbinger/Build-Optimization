"""
Selection-and-Construction Policies Package.

This package contains algorithms that perform **joint** mandatory-bin selection
and route construction in a single integrated optimisation loop, rather than
executing the two phases sequentially.
"""

from .base import (
    BaseJointPolicy,
    JointPolicyFactory,
    JointPolicyRegistry,
)
from .joint_greedy_orienteering import JointGreedyParams, JointGreedyPolicy
from .joint_simulated_annealing import JointSAParams, JointSAPolicy
from .non_dominated_sorting_biased_random_key_genetic_algorithm import (
    NDSBRKGAParams,
    NDSBRKGAPolicy,
)

__all__ = [
    "BaseJointPolicy",
    "JointPolicyRegistry",
    "JointPolicyFactory",
    "NDSBRKGAPolicy",
    "NDSBRKGAParams",
    "JointSAPolicy",
    "JointSAParams",
    "JointGreedyPolicy",
    "JointGreedyParams",
]

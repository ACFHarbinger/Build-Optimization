"""
Modular loss functions for imitation learning and policy optimization.

This module provides reusable loss functions that can be composed
to build custom training objectives for RL and IL algorithms.
"""

from .js_divergence_loss import js_divergence_loss
from .kl_divergence_loss import kl_divergence_loss, reverse_kl_divergence_loss
from .nll_loss import nll_loss
from .weighted_nll_loss import weighted_nll_loss

__all__ = [
    "js_divergence_loss",
    "kl_divergence_loss",
    "nll_loss",
    "reverse_kl_divergence_loss",
    "weighted_nll_loss",
]

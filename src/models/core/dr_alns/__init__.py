"""
DR-ALNS (Deep Reinforcement Learning - Adaptive Large Neighborhood Search).

Hybrid approach combining PPO-based deep RL with ALNS for online control
of operator selection and parameter configuration.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from .dr_alns_solver import DRALNSSolver
from .ppo_agent import DRALNSPPOAgent, DRALNSState
from .ppo_trainer import PPOTrainer

__all__ = ["DRALNSSolver", "DRALNSPPOAgent", "DRALNSState", "PPOTrainer"]

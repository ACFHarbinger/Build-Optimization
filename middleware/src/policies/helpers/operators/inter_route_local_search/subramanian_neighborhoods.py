"""
Subramanian Inter-Route Operators Module.

This module implements the specific inter-route neighborhoods used in:
    Subramanian et al. (2013) "A hybrid algorithm for a class of vehicle routing problems"

Neighborhoods:
- Shift(2,0): Relocate 2 consecutive nodes.
- Swap(2,1): Swap 2 consecutive nodes with 1 node.
- Swap(2,2): Swap 2 consecutive nodes with 2 consecutive nodes.
- Cross: Exchange route suffixes (tails).
"""

from typing import Any

from .exchange_chain import exchange_k_0, exchange_k_h
from .k_opt_star import move_kopt_star


def shift_2_0(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Shift(2,0): Relocate 2 consecutive nodes from r_src to r_dst.
    Inserted after pos_dst.
    """
    return exchange_k_0(ls, r_src, pos_src, r_dst, pos_dst, k=2)


def swap_2_1(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Swap(2,1): Swap 2 consecutive nodes from r_src with 1 node from r_dst.
    """
    return exchange_k_h(ls, r_src, pos_src, 2, r_dst, pos_dst, 1)


def swap_2_2(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Swap(2,2): Swap 2 consecutive nodes from r_src with 2 consecutive nodes from r_dst.
    """
    return exchange_k_h(ls, r_src, pos_src, 2, r_dst, pos_dst, 2)


def move_cross(ls: Any, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    Cross: Exchange suffixes of routes r_u and r_v.
    Suffix of r_u starts after p_u; suffix of r_v starts after p_v.
    This is equivalent to 2-opt* (move_kopt_star with 2 cuts).
    """
    return move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v)])

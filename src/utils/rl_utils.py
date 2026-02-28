"""
Reinforcement Learning utilities for WSmart-Route.
"""

from typing import Any

import torch
from tensordict import TensorDict

from interfaces import ITensorDictLike, ITraversable


def _internal_safe_copy(obj: object, visited: set):
    """Internal helper for safe_td_copy to handle recursion and avoid circular refs."""
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    try:
        if isinstance(obj, torch.Tensor):
            return obj.clone()

        if isinstance(obj, ITensorDictLike):
            clean_dict = {}
            for k, v in obj.items():
                if isinstance(v, (torch.Tensor, dict, ITensorDictLike)):
                    copied_v = _internal_safe_copy(v, visited)
                    if copied_v is not None:
                        clean_dict[k] = copied_v
            return TensorDict(clean_dict, batch_size=obj.batch_size, device=obj.device)

        if isinstance(obj, ITraversable):
            res_dict = {}
            for k, v in obj.items():
                copied_v = _internal_safe_copy(v, visited)
                if copied_v is not None:
                    res_dict[k] = copied_v
            return res_dict

        if isinstance(obj, (list, tuple)):
            res_list = [v for x in obj if (v := _internal_safe_copy(x, visited)) is not None]
            return type(obj)(res_list)

        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
    except Exception:
        return None
    return None


def safe_td_copy(td: object) -> Any:
    """
    Perform a deep copy of a TensorDict that is safe against circular references
    and complex objects.
    """
    if not isinstance(td, ITensorDictLike):
        return td

    res = _internal_safe_copy(td, set())
    if res is None:
        return TensorDict({}, batch_size=td.batch_size, device=td.device)
    return res
